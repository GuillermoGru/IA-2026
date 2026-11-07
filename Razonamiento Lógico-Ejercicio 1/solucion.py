from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

PATRON_ATOMO = re.compile(r"^([A-Za-z_][\w]*)\s*(?:\((.*)\))?\s*$")


@dataclass(frozen=True)
class Atomo:
    predicado: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.predicado
        return f"{self.predicado}({', '.join(self.args)})"


@dataclass(frozen=True)
class Regla:
    id_regla: str
    premisas: tuple[Atomo, ...]
    cabeza: Atomo

    def __str__(self) -> str:
        cuerpo = " ∧ ".join(str(p) for p in self.premisas) if self.premisas else "true"
        return f"{self.id_regla}: {cuerpo} → {self.cabeza}"


@dataclass
class BaseConocimiento:
    hechos: list[Atomo]
    reglas: list[Regla]
    consulta: Atomo | None = None
    origen: Path | None = None

    def copiar(self) -> BaseConocimiento:
        return BaseConocimiento(list(self.hechos), list(self.reglas), self.consulta, self.origen)


@dataclass
class ResultadoInferencia:
    probada: bool
    sustitucion: dict[str, str] | None
    atomos_registrados: list[Atomo]
    traza: list[str] = field(default_factory=list)


def es_variable(termino: str) -> bool:
    return bool(termino) and termino[0].islower()


def parsear_atomo(texto: str) -> Atomo:
    texto = texto.strip()
    coincidencia = PATRON_ATOMO.match(texto)
    if not coincidencia:
        raise ValueError(f"No se pudo parsear el átomo: {texto!r}")
    predicado, args_crudos = coincidencia.group(1), coincidencia.group(2)
    if args_crudos is None or args_crudos.strip() == "":
        return Atomo(predicado, ())
    args = tuple(parte.strip() for parte in args_crudos.split(","))
    if any(not parte for parte in args):
        raise ValueError(f"Argumento vacío en el átomo: {texto!r}")
    return Atomo(predicado, args)


def _parsear_yaml_simple(texto: str) -> dict[str, Any]:
    hechos: list[str] = []
    reglas: list[dict[str, Any]] = []
    consulta: str | None = None
    seccion: str | None = None
    regla_actual: dict[str, Any] | None = None

    for linea in texto.splitlines():
        limpia = linea.strip()
        if not limpia:
            continue
        if limpia == "facts:":
            seccion = "facts"
            continue
        if limpia == "rules:":
            seccion = "rules"
            continue
        if limpia.startswith("query:"):
            consulta = limpia.split(":", 1)[1].strip().strip("\"'")
            seccion = None
            continue
        if seccion == "facts" and limpia.startswith("- "):
            hechos.append(limpia[2:].strip().strip("\"'"))
        elif seccion == "rules":
            if limpia.startswith("- id:"):
                regla_actual = {"id": limpia.split(":", 1)[1].strip(), "if": [], "then": ""}
                reglas.append(regla_actual)
            elif regla_actual is not None and limpia.startswith("if:"):
                contenido = limpia.split(":", 1)[1].strip()
                if contenido.startswith("[") and contenido.endswith("]"):
                    items = [x.strip().strip("\"'") for x in contenido[1:-1].split(",") if x.strip()]
                    regla_actual["if"] = items
            elif regla_actual is not None and limpia.startswith("then:"):
                regla_actual["then"] = limpia.split(":", 1)[1].strip().strip("\"'")

    return {"facts": hechos, "rules": reglas, "query": consulta}


def cargar_kb(ruta: str | Path) -> BaseConocimiento:
    ruta = Path(ruta)
    contenido = ruta.read_text(encoding="utf-8")
    if yaml is not None:
        datos: dict[str, Any] = yaml.safe_load(contenido) or {}
    else:
        datos = _parsear_yaml_simple(contenido)
    hechos = [parsear_atomo(item) for item in (datos.get("facts") or [])]
    reglas: list[Regla] = []
    for idx, item in enumerate(datos.get("rules") or [], start=1):
        id_regla = str(item.get("id") or f"R{idx}")
        premisas = tuple(parsear_atomo(p) for p in (item.get("if") or []))
        cabeza = parsear_atomo(item["then"])
        reglas.append(Regla(id_regla, premisas, cabeza))
    consulta_raw = datos.get("query")
    consulta = parsear_atomo(consulta_raw) if consulta_raw else None
    return BaseConocimiento(hechos, reglas, consulta, ruta)


def aplicar_termino(termino: str, subst: dict[str, str]) -> str:
    visitados: set[str] = set()
    while es_variable(termino) and termino in subst and termino not in visitados:
        visitados.add(termino)
        termino = subst[termino]
    return termino


def aplicar_atomo(atomo: Atomo, subst: dict[str, str]) -> Atomo:
    return Atomo(atomo.predicado, tuple(aplicar_termino(a, subst) for a in atomo.args))


def _ocurre(var: str, termino: str, subst: dict[str, str]) -> bool:
    termino = aplicar_termino(termino, subst)
    return var == termino


def unificar(
    a: Atomo | str,
    b: Atomo | str,
    subst: dict[str, str] | None = None,
) -> dict[str, str] | None:
    if subst is None:
        subst = {}
    else:
        subst = dict(subst)

    if isinstance(a, Atomo) and isinstance(b, Atomo):
        if a.predicado != b.predicado or len(a.args) != len(b.args):
            return None
        for izq, der in zip(a.args, b.args):
            subst = unificar(izq, der, subst)
            if subst is None:
                return None
        return subst

    if isinstance(a, Atomo) or isinstance(b, Atomo):
        return None

    a = aplicar_termino(a, subst)
    b = aplicar_termino(b, subst)
    if a == b:
        return subst
    if es_variable(a):
        if _ocurre(a, b, subst):
            return None
        subst[a] = b
        return subst
    if es_variable(b):
        if _ocurre(b, a, subst):
            return None
        subst[b] = a
        return subst
    return None


def formatear_subst(subst: dict[str, str]) -> str:
    if not subst:
        return "{}"
    partes = ", ".join(f"{k}/{aplicar_termino(k, subst)}" for k in sorted(subst.keys()))
    return "{" + partes + "}"


def _estandarizar_variables(regla: Regla, contador: int) -> Regla:
    mapa: dict[str, str] = {}

    def renombrar(atomo: Atomo) -> Atomo:
        nuevos_args = []
        for arg in atomo.args:
            if es_variable(arg):
                mapa.setdefault(arg, f"{arg}_{contador}")
                nuevos_args.append(mapa[arg])
            else:
                nuevos_args.append(arg)
        return Atomo(atomo.predicado, tuple(nuevos_args))

    return Regla(
        regla.id_regla,
        tuple(renombrar(p) for p in regla.premisas),
        renombrar(regla.cabeza),
    )


def _satisfacer_premisas(
    premisas: tuple[Atomo, ...],
    hechos: list[Atomo],
    subst: dict[str, str],
) -> list[dict[str, str]]:
    if not premisas:
        return [subst]
    primera, resto = premisas[0], premisas[1:]
    meta = aplicar_atomo(primera, subst)
    resultados: list[dict[str, str]] = []
    for hecho in hechos:
        theta = unificar(meta, hecho, subst)
        if theta is not None:
            resultados.extend(_satisfacer_premisas(resto, hechos, theta))
    return resultados


def encadenamiento_adelante(
    kb: BaseConocimiento,
    consulta: Atomo,
    *,
    max_rondas: int = 50,
) -> ResultadoInferencia:
    hechos = list(kb.hechos)
    derivados: list[Atomo] = []
    traza = [
        f"Hechos iniciales: {', '.join(str(h) for h in hechos) or '(ninguno)'}",
        f"Consulta: {consulta}",
    ]

    contador = 0
    for num_ronda in range(1, max_rondas + 1):
        nuevos_atomos: list[Atomo] = []
        traza.append(f"--- Ronda {num_ronda} ---")
        for regla in kb.reglas:
            contador += 1
            regla_fresca = _estandarizar_variables(regla, contador)
            for theta in _satisfacer_premisas(regla_fresca.premisas, hechos, {}):
                cabeza = aplicar_atomo(regla_fresca.cabeza, theta)
                if cabeza in hechos or cabeza in nuevos_atomos:
                    continue
                nuevos_atomos.append(cabeza)
                derivados.append(cabeza)
                traza.append(f"Dispara {regla.id_regla}  {formatear_subst(theta)}  ⇒  {cabeza}")
        if not nuevos_atomos:
            traza.append("Sin hechos nuevos (punto fijo alcanzado).")
            break
        hechos.extend(nuevos_atomos)
    else:
        traza.append("Detenido: se alcanzó max_rondas.")

    for hecho in hechos:
        theta = unificar(consulta, hecho, {})
        if theta is not None:
            traza.append(f"Consulta probada: {consulta}")
            return ResultadoInferencia(True, theta, derivados, traza)

    traza.append(f"Consulta NO probada: {consulta}")
    return ResultadoInferencia(False, None, derivados, traza)


def encadenamiento_atras(
    kb: BaseConocimiento,
    consulta: Atomo,
    *,
    max_profundidad: int = 25,
) -> ResultadoInferencia:
    traza = [f"Consulta: {consulta}"]
    contador = [0]
    metas_visitadas: list[Atomo] = []

    def sangria(profundidad: int) -> str:
        return "  " * profundidad

    def bc_or(
        meta: Atomo,
        subst: dict[str, str],
        profundidad: int,
        pila: tuple[str, ...],
    ) -> dict[str, str] | None:
        meta = aplicar_atomo(meta, subst)
        clave = str(meta)
        if clave in pila:
            traza.append(f"{sangria(profundidad)}ciclo: {meta} ya está en la pila de metas — se omite")
            return None
        if profundidad > max_profundidad:
            traza.append(f"{sangria(profundidad)}profundidad máxima en {meta}")
            return None

        traza.append(f"{sangria(profundidad)}meta   {meta}")
        metas_visitadas.append(meta)

        for hecho in kb.hechos:
            theta = unificar(meta, hecho, subst)
            if theta is not None:
                traza.append(f"{sangria(profundidad)}hecho  {hecho}  {formatear_subst(theta)}")
                return theta

        for regla in kb.reglas:
            contador[0] += 1
            regla_fresca = _estandarizar_variables(regla, contador[0])
            theta = unificar(meta, regla_fresca.cabeza, subst)
            if theta is None:
                continue
            traza.append(f"{sangria(profundidad)}regla  {regla.id_regla}  {formatear_subst(theta)}")
            respuesta = bc_and(regla_fresca.premisas, theta, profundidad + 1, pila + (clave,))
            if respuesta is not None:
                return respuesta

        traza.append(f"{sangria(profundidad)}fallo  {meta}")
        return None

    def bc_and(
        premisas: tuple[Atomo, ...],
        subst: dict[str, str],
        profundidad: int,
        pila: tuple[str, ...],
    ) -> dict[str, str] | None:
        if not premisas:
            return subst
        primera, resto = premisas[0], premisas[1:]
        theta = bc_or(primera, subst, profundidad, pila)
        if theta is None:
            return None
        return bc_and(resto, theta, profundidad, pila)

    encontrada = bc_or(consulta, {}, 0, ())
    probada = encontrada is not None
    traza.append(f"Consulta probada: {consulta}" if probada else f"Consulta NO probada: {consulta}")

    metas_unicas: list[Atomo] = []
    vistas: set[str] = set()
    for m in metas_visitadas:
        if str(m) not in vistas:
            vistas.add(str(m))
            metas_unicas.append(m)

    return ResultadoInferencia(probada, encontrada, metas_unicas, traza)


def imprimir_kb(kb: BaseConocimiento) -> None:
    print("Base de Conocimiento")
    print("=" * 60)
    if kb.origen:
        print(f"Archivo: {kb.origen.name}")
    print("Hechos:")
    for h in kb.hechos:
        print(f"  - {h}")
    print("Reglas:")
    for r in kb.reglas:
        print(f"  - {r}")
    print()


def ejecutar_consulta(kb: BaseConocimiento, texto_consulta: str) -> None:
    consulta = parsear_atomo(texto_consulta)
    res_fc = encadenamiento_adelante(kb, consulta)
    res_bc = encadenamiento_atras(kb, consulta)

    print("=" * 60)
    print(f"CONSULTA: {consulta}")
    print("=" * 60)

    print("\n[1] Encadenamiento hacia adelante (Forward Chaining)")
    print("-" * 60)
    for linea in res_fc.traza:
        print(linea)
    print(f"Átomos nuevos derivados ({len(res_fc.atomos_registrados)}): "
          f"{', '.join(str(a) for a in res_fc.atomos_registrados)}")
    print(f"¿Probada?: {'SÍ' if res_fc.probada else 'NO'}")

    print("\n[2] Encadenamiento hacia atrás (Backward Chaining)")
    print("-" * 60)
    for linea in res_bc.traza:
        print(linea)
    print(f"Metas visitadas ({len(res_bc.atomos_registrados)}): "
          f"{', '.join(str(a) for a in res_bc.atomos_registrados)}")
    print(f"¿Probada?: {'SÍ' if res_bc.probada else 'NO'}")
    print()


def main() -> None:
    ruta_por_defecto = Path(__file__).resolve().parent / "biblioteca.yaml"
    parser = argparse.ArgumentParser(
        description="Encadenamiento hacia adelante y hacia atrás en la biblioteca."
    )
    parser.add_argument("--kb", type=Path, default=ruta_por_defecto, help="Ruta al archivo YAML")
    parser.add_argument("--query", type=str, default=None, help="Consulta específica a evaluar")
    args = parser.parse_args()

    kb = cargar_kb(args.kb)
    imprimir_kb(kb)

    if args.query:
        ejecutar_consulta(kb, args.query)
        return

    consultas = [
        "AccesoNoche(Ana)",
        "AccesoNoche(Luis)",
        "AccesoNoche(Marta)",
    ]
    for q in consultas:
        ejecutar_consulta(kb, q)

    print("#" * 60)
    print("EXPERIMENTO ADICIONAL (Pregunta 3 del reporte):")
    print("Añadiendo el hecho TieneCredencial(Luis) y consultando AccesoNoche(Luis)")
    print("#" * 60)
    kb_luis = kb.copiar()
    kb_luis.hechos.append(parsear_atomo("TieneCredencial(Luis)"))
    ejecutar_consulta(kb_luis, "AccesoNoche(Luis)")


if __name__ == "__main__":
    main()
