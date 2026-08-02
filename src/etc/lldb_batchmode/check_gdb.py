import sys
import traceback

import gdb

from .common import (
    BLESS,
    INPUT_DATA,
    TYPES_TESTED,
    VARS_TESTED,
    Child,
    Result,
    Variable,
    print_error,
    print_mismatch,
)
from .from_gdb import (
    TypeCode,
    bless_variable,
    get_breakpoint_idx,
    get_children,
    get_fields,
    get_template_args,
    get_type_name,
    make_visualizer,
    type_from_gdb,
    variable_from_gdb,
)


def check(var_name: str):
    if BLESS:
        print(f"blessing var {var_name}")
        bless_variable(var_name)

    # Even if we're blessing, we still want to run the variable through the test to make sure we're
    # not somehow saving invalid information

    valobj = gdb.selected_frame().read_var(var_name)

    var = variable_from_gdb(valobj)

    breakpoint_idx = get_breakpoint_idx()

    try:
        expected = INPUT_DATA.breakpoints[breakpoint_idx][var_name]
    except IndexError:
        print_error("INPUT_DATA", f"No data found for breakpoint #{breakpoint_idx}")
        return Result.Mismatch
    except KeyError:
        print_error(
            "INPUT_DATA",
            f"No data found for var '{var_name}' at breakpoint #{breakpoint_idx}",
        )
        return Result.Mismatch

    result = var_matches(var, expected, valobj, var_name)
    # --bless outputs blank breakpoints for any breakpoints with no variables, so we need to account
    # for that here
    if len(VARS_TESTED) <= breakpoint_idx:
        VARS_TESTED.extend({} for _ in range(1 + breakpoint_idx - len(VARS_TESTED)))

    VARS_TESTED[breakpoint_idx][var_name] = result

    if result == Result.Ok:
        print(f"{var_name}: Ok")

    return result


def type_matches(gdb_type: gdb.Type, provider_ok: bool = False):
    name = get_type_name(gdb_type)
    error_source = f"type '{name}'"

    if (r := TYPES_TESTED.get(name)) is not None:
        # The proper result was returned the first time the type was tested, so we can just pretend
        # everything we've already seen has succeeded.
        if not r:
            print_error(
                f"type '{name}'", f"mismatch (see prior output for type '{name}')"
            )
        return r

    ty = type_from_gdb(gdb_type)

    expected = INPUT_DATA.types.get(name)

    if expected is None:
        result = Result.Mismatch
        print_error(f"type '{name}'", "type not found in input data")
    else:
        basic_type_result = (
            Result.Ok if ty.basic_type == expected.basic_type else Result.Mismatch
        )

        type_class_result = (
            Result.Ok if ty.type_class == expected.type_class else Result.Mismatch
        )

        if type_class_result == Result.Mismatch:
            print_mismatch(
                error_source,
                "type_class (gdb.TYPE_CODE_)",
                f"{ty.type_class} ({TypeCode(ty.type_class).name})",
                f"{expected.type_class} ({TypeCode(expected.type_class).name})",
            )

        ty_result = ty.matches(expected, name, provider_ok)

        result = basic_type_result and type_class_result and ty_result

    TYPES_TESTED[name] = result

    fields = get_fields(gdb_type)
    inner_types = [f.type for f in fields]

    inner_types.extend(get_template_args(gdb_type))

    for t in inner_types:
        result = type_matches(t) and result

    return result


def var_matches(
    var: Variable, expected: Variable, valobj: gdb.Value, var_name: str
) -> Result:
    # Happy path requires very little intercession from us. We keep these values on the stack
    # so we don't have to recalculate them if we need to do error handling
    summary_ok = var.summary == expected.summary
    synthetic_ok = var.synthetic == expected.synthetic
    pretty_type_name_ok = var.pretty_type_name == expected.pretty_type_name
    pretty_print_ok = var.pretty_print == expected.pretty_print
    format_ok = var.format == expected.format

    type_ok = var.type == expected.type

    if var.has_visualizer() or expected.has_visualizer():
        type_match_ok = type_matches(
            valobj.type,
            summary_ok
            and synthetic_ok
            and format_ok
            and pretty_type_name_ok
            and pretty_print_ok,
        )
    else:
        type_match_ok = Result.Ok

    value_ok = var.value == expected.value

    work_list = [c for _i, c in get_children(valobj)]
    child_types_ok = True

    while len(work_list) != 0:
        child = work_list.pop()
        work_list.extend([c for _i, c in get_children(child)])

        child_types_ok &= type_matches(child.type) == Result.Ok

    children_ok = children_match(var.children, expected.children, var_name, valobj)

    if (
        type_ok
        and type_match_ok
        and pretty_type_name_ok
        and pretty_print_ok
        and value_ok
        and synthetic_ok
        and summary_ok
        and format_ok
        and children_ok
        and child_types_ok
    ):
        return Result.Ok

    error_source = f"var '{var_name}'"

    # otherwise, we want to output exactly what doesn't match
    # and any additional helpful information

    # We check the type first. If this has changed, it's relatively likely nothing else will work
    # properly
    if not type_ok:
        print_mismatch(
            error_source,
            "type (Type Name)",
            var.type,
            expected.type,
        )

    # We check the summary next since it's the most user-visible output. We don't need to check
    # `pretty_print` if the summary provider doesn't match.
    if not summary_ok:
        print_mismatch(
            error_source,
            "summary (PrettyPrinter.to_string)",
            var.summary,
            expected.summary,
        )
    elif not pretty_print_ok:
        print_mismatch(
            error_source,
            "pretty_print (Summary Output)",
            var.pretty_print,
            expected.pretty_print,
        )

        # try the summary provider directly to see if it's throwing an exception
        if var.summary is not None:
            try:
                provider = make_visualizer(valobj)
                _ = provider.to_string()
            except Exception as e:
                print_error(
                    error_source + " Summary",
                    "Error while running Summary \
provider:",
                )
                traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)

    # Next we check the value and formatter. These mostly affect primitives.
    if not value_ok:
        print_mismatch(error_source, "value", var.value, expected.value)
    if not format_ok:
        print_mismatch(error_source, "format", var.format, expected.format)

    # Synthetic is checked next since children, pretty type name, and pretty print rely on it. If
    # the synthetic doesn't match, we can assume those won't match either.
    if not synthetic_ok:
        print_mismatch(
            error_source,
            "synthetic (PrettyPrinter)",
            var.synthetic,
            expected.synthetic,
        )
    else:
        if not pretty_type_name_ok:
            print_mismatch(
                error_source,
                f"pretty_type_name ({var.synthetic}.get_type_name)",
                var.pretty_type_name,
                expected.pretty_type_name,
            )

        if not children_ok and var.synthetic is not None:
            try:
                # check for exceptions in the initializer
                _synth = make_visualizer(valobj)
                # FIXME(Walnut356) at the moment I haven't fiddled with GDB enough to know what the
                # common failure states are for their pretty printers, so at the moment this check
                # is pretty barebones
            except Exception as e:
                print_error(
                    error_source + " Synthetic",
                    "Error while running Synthetic\
Provider:",
                )
                traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)

    return Result.Mismatch


def children_match(
    children: list[Child],
    expected: list[Child],
    path: str,
    valobj: gdb.Value,
) -> Result:
    result = Result.Ok if len(children) == len(expected) else Result.Mismatch

    mismatches = []
    missing = []

    valobj_children = get_children(make_visualizer(valobj))

    for i in range(len(expected)):
        exp = expected[i]

        if i >= len(children):
            missing.append(exp.name)
            continue

        got = children[i]

        if got.name != exp.name or got.type != exp.type or got.value != exp.value:
            result = Result.Mismatch
            mismatches.append(
                f"{exp.name}: {exp.type} = {exp.value} -> {got.name}: {got.type} = {got.value}"
            )
        # no point recursing into children if we've already mismatched
        elif len(exp.children) != 0:
            result &= children_match(
                got.children,
                exp.children,
                f"{path}.{exp.name}",
                valobj_children[0][1],
            )

    if result == Result.Ok:
        return result

    if len(mismatches) != 0:
        error_str = "\n    ".join(mismatches)
        print_error(
            path,
            f"The following children do not match (expected -> got):\n    {error_str}",
        )
    elif len(missing) != 0:
        error_str = ", ".join(missing)
        print_error(
            path,
            f"The following children were expected, but were not found:\n    {error_str}",
        )
    elif len(children) > len(expected):
        error_str = "\n    ".join(
            f"{got.name}: {got.type} = {got.value}" for got in children[len(expected) :]
        )
        print_error(
            path,
            f"The following children were found, but were not expected:\n    {error_str}",
        )

    return result
