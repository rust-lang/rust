from enum import Enum
from types import ModuleType
from typing import Callable, List, Tuple

import gdb

from ..common import (
    INPUT_DATA,
    Child,
    Field,
    Type,
    Variable,
)
import importlib.util
import sys


# For whatever reason, gdb doesn't export several of the modules it distributes. It adds its own
# top level module to the search path, but that doesn't let us import the unexported parts inside
# the `gdb` package. To sidestep that, we can manually invoke Python's import pipeline on the
# desired file.
def import_from_path(module_name, file_path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gdb_printing = import_from_path("gdb_printing", gdb.PYTHONDIR + "/gdb/printing.py")

make_visualizer: Callable[[gdb.Value], "gdb._PrettyPrinter"] = (
    gdb_printing.make_visualizer
)
"""If a pretty printer exists that handles this value's type, returns the pretty printer
instantiated with the given value. If none can be found, returns a "NoOp" pretty printer with the
same interface as a custom pretty printer.
"""


# Same trick as from_lldb.TypeClass/from_lldb.BasicType
_gdb_type_codes = {
    k.removeprefix("TYPE_CODE_"): v
    for k, v in gdb.__dict__.items()
    if k.startswith("TYPE_CODE_")
}


class TypeCode(Enum):
    """Direct mapping of `gdb.TYPE_CODE_` enum for convenience. Used to print a more meaningful
    error message when Type.type_class does not match.
    """

    vars().update(_gdb_type_codes)


def get_template_args(ty: gdb.Type) -> List[gdb.Type]:
    template_args = []
    i = 0
    while True:
        # AFAIK gdb does not have functionality to get the number of template args ahead of time, so
        # we just have to iterate until an exception is thrown.
        try:
            template_args.append(ty.template_argument(i))
            i += 1
        except Exception:
            break

    return template_args


def get_fields(ty: gdb.Type) -> List[gdb.Field]:
    # per GDB docs:

    # "Return the fields of this type. The behavior depends on the type code:
    # For structure and union types, this method returns the fields.
    # Enum types have one field per enum constant.
    # Function and method types have one field per parameter. The base types
    # of C++ classes are also represented as fields.
    # Array types have one field representing the array’s range.
    # If the type does not fit into one of these categories, a TypeError is # raised."
    try:
        return ty.fields()
    except TypeError:
        return []


def get_type_name(ty: gdb.Type) -> str:
    """Attempts all possible ways of acquiring a type's name, returning the first non-None value."""

    # It seems that when GDB's built-in rust handling overrides a type name (e.g. pointers and
    # refs -> `*mut T`), the `Type.name` field is set to `None`. In this case, we can still recover
    # the overridden name via `str(ty)`.
    return ty.name or ty.tag or str(ty) or "<unable to read type name>"


def type_from_gdb(ty: gdb.Type) -> Type:
    return Type(
        ty.sizeof,
        # maybe set basic type == is_scalar instead?
        0,
        ty.code,
        [field_from_gdb(f) for f in get_fields(ty)],
        [arg.name for arg in get_template_args(ty)],
    )


def field_from_gdb(field: gdb.Field) -> Field:
    return Field(field.name, field.type.name, field.bitpos // 8)


def variable_from_gdb(var: gdb.Value) -> Variable:
    ty = var.type
    ty_name = get_type_name(ty)

    pretty_type_name = None

    for printer in gdb.type_printers:
        if not printer.enabled:
            continue
        pretty_type_name = printer.instantiate().recognize(ty)

        if pretty_type_name is not None:
            break

    not_ptr = ty.is_scalar and ty.code not in (
        gdb.TYPE_CODE_PTR,
        gdb.TYPE_CODE_REF,
        gdb.TYPE_CODE_RVALUE_REF,
    )

    if not_ptr:
        value = var.format_string(raw=True)
    else:
        value = None

    # Returns either the registered visualizer, or a `NoOp` visualizer that
    # implements default behavior. We later use the "default" visualizer for
    # its `.children()`
    visualizer = make_visualizer(var)

    format = None

    if type(visualizer).__name__.startswith("NoOp"):
        synthetic = None
        summary = None
    else:
        format = None
        synthetic = type(visualizer).__name__
        if getattr(visualizer, "to_string", None) is not None:
            summary = synthetic + ".to_string"
        else:
            summary = None
        # Maybe:
        # format = gdb.print_options()?

    pretty_print = var.format_string()

    children = [child_from_gdb(i, c) for i, c in get_children(visualizer)]

    return Variable(
        ty_name,
        pretty_type_name,
        pretty_print,
        value,
        synthetic,
        summary,
        format,
        children,
    )


def get_children(obj) -> List[Tuple[str, gdb.Value]]:
    children = getattr(obj, "children", None)

    if children is None:
        return []

    return children()


def child_from_gdb(ident: str, child: gdb.Value) -> Child:
    ty = child.type
    if ty.is_scalar and ty.code not in (
        gdb.TYPE_CODE_PTR,
        gdb.TYPE_CODE_REF,
        gdb.TYPE_CODE_RVALUE_REF,
    ):
        value = child.format_string()
    else:
        value = None

    visualizer = make_visualizer(child)

    return Child(
        ident,
        get_type_name(ty),
        value,
        [child_from_gdb(i, c) for i, c in get_children(visualizer)],
    )


def get_breakpoint_idx() -> int:
    bp_idx = 0

    for i, bp in enumerate(gdb.breakpoints()):
        if bp.hit_count != 0:
            bp_idx = i
        else:
            break

    return bp_idx


def bless_variable(var_name: str):
    value = gdb.selected_frame().read_var(var_name)
    var_data = variable_from_gdb(value)

    breakpoint_idx = get_breakpoint_idx()

    if len(INPUT_DATA.breakpoints) <= breakpoint_idx:
        INPUT_DATA.breakpoints.extend(
            {} for i in range(1 + breakpoint_idx - len(INPUT_DATA.breakpoints))
        )

    INPUT_DATA.breakpoints[breakpoint_idx][var_name] = var_data

    # Don't bless types if we don't have anything that could possibly break from the type changing
    if not var_data.has_visualizer():
        return

    work_list = [value]
    while len(work_list) != 0:
        val = work_list.pop()
        obj = make_visualizer(val)
        children = getattr(obj, "children", None)
        if children is not None:
            work_list.extend([c for _n, c in children()])

        bless_type(val.type)


def bless_type(ty: gdb.Type):
    name = get_type_name(ty)
    data = type_from_gdb(ty)

    if name in INPUT_DATA.types:
        import pprint

        assert (
            INPUT_DATA.types[name] == data
        ), f"old: {pprint.pformat(INPUT_DATA.types[name])}\nnew: {pprint.pformat(data)}"

        return

    print(f"blessing type: {name}")

    INPUT_DATA.types[name] = data

    try:
        for f in ty.fields():
            if f.type is not None:
                bless_type(f.type)
    except TypeError:
        pass

    i = 0
    while True:
        # AFAIK gdb does not have functionality to get the number of template args ahead of time, so
        # we just have to iterate until an exception is thrown.
        try:
            arg = ty.template_argument(i)
            i += 1

            bless_type(arg)
        except Exception:
            break
