"""Contains the logic that compares variables to `INPUT_DATA` via the entrypoint
`check(var_name, breakpoint_idx, frame)`. These comparisons report errors to stdout, and then return
a `Result` indicating whether or not the variable matched.

Checks *do not* stop after the first encountered error. Some redundant information may be ommitted
(e.g. checking pretty printed type name if the synthetic isn't properly attached to the type).
"""

from typing import Any, Callable
import traceback
import sys

import lldb
from .common import (
    BLESS,
    INPUT_DATA,
    Child,
    Variable,
    Result,
    print_error,
    print_mismatch,
)
from .from_lldb import (
    BasicType,
    TypeClass,
    bless_variable,
    variable_from_lldb,
    type_from_lldb,
    get_generics,
)


VARS_TESTED: list[dict[str, Result]] = []
"""Used to help ensure all expected variables were tested. Each element of the list corresponds to a
breakpoint, and contains a set of all of the variable names tested for that breakpoint."""


def check(var_name: str, breakpoint_idx: int, frame: lldb.SBFrame) -> Result:
    """`lldb-repr` pseudo-command entrypoint. Checks the variable against `INPUT_DATA` for the given
    frame at the given breakpoint.
    """

    if BLESS:
        print(f"blessing var {var_name}")
        bless_variable(INPUT_DATA, var_name, breakpoint_idx, frame)

    # Even if we're blessing, we still want to run the variable through the test to make sure we're
    # not somehow saving invalid information

    valobj: lldb.SBValue = frame.var(var_name)
    if not valobj.IsValid():
        print_error(var_name, "Unable to find variable")
        return Result.Mismatch

    var = variable_from_lldb(valobj)

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

    result = var_matches(var, expected, valobj)
    if len(VARS_TESTED) <= breakpoint_idx:
        VARS_TESTED.append({})

    VARS_TESTED[breakpoint_idx][var_name] = result

    if result == Result.Ok:
        print(f"{var_name}: Ok")

    return result


TYPES_TESTED: dict[str, Result] = {}
"""Since types are unique and unchanging, we only need to test each type once. This also helps
ensure we have tested all types in `INPUT_DATA`
"""


def type_matches(
    sbtype: lldb.SBType, sbtarget: lldb.SBTarget, provider_ok: bool = False
) -> Result:
    """Checks a type and all field/generic types (recursively) against the data contained in
    `INPUT_DATA`."""
    name: str = sbtype.GetName()
    error_source = f"type '{name}'"
    # print(f"  checking type: {name}")

    if (r := TYPES_TESTED.get(name)) is not None:
        # The proper result was returned the first time the type was tested, so we can just pretend
        # everything we've already seen has succeeded.
        if not r:
            print_error(
                f"type '{name}'", f"mismatch (see prior output for type '{name}')"
            )
        return r

    ty = type_from_lldb(sbtype, sbtarget)

    expected = INPUT_DATA.types.get(name)

    if expected is None:
        result = Result.Mismatch
        print_error(f"type '{name}'", "type not found in input data")
    else:
        basic_type_result = (
            Result.Ok if ty.basic_type == expected.basic_type else Result.Mismatch
        )
        if basic_type_result == Result.Mismatch:
            print_mismatch(
                error_source,
                "basic_type (lldb.eBasicType)",
                f"{ty.basic_type} ({BasicType(ty.basic_type)})",
                f"{expected.basic_type} ({BasicType(expected.basic_type)})",
            )

        type_class_result = (
            Result.Ok if ty.type_class == expected.type_class else Result.Mismatch
        )

        if type_class_result == Result.Mismatch:
            print_mismatch(
                error_source,
                "type_class (lldb.eTypeClass)",
                f"{ty.type_class} ({TypeClass(ty.type_class).name})",
                f"{expected.type_class} ({TypeClass(expected.type_class).name})",
            )

        ty_result = ty.matches(expected, name, provider_ok)

        result = basic_type_result and type_class_result and ty_result

    TYPES_TESTED[name] = result

    fields: list[lldb.SBTypeMember] = sbtype.fields
    inner_types = [f.GetType() for f in fields]
    inner_types.extend(get_generics(sbtype, sbtarget))

    for t in inner_types:
        result = type_matches(t, sbtarget) and result

    return result


def tested_all_types() -> bool:
    """Returns true if all types in INPUT_DATA were tested this run."""

    expected_types = set(k for k in INPUT_DATA.types)
    untested_types = expected_types.difference(TYPES_TESTED.keys())

    if len(untested_types) != 0:
        print(
            f"[repr error] The following types were expected, but were not tested:\n\
  {untested_types}"
        )

    return len(untested_types) == 0


def tested_all_variables() -> bool:
    expected_vars = [set(k for k in vars) for vars in INPUT_DATA.breakpoints]
    untested_vars = [
        expected.difference(tested.keys())
        for expected, tested in zip(expected_vars, VARS_TESTED)
    ]

    tested_not_expected = [
        set(tested.keys()).difference(expected)
        for expected, tested in zip(expected_vars, VARS_TESTED)
    ]

    result = True

    for i, v in enumerate(untested_vars):
        if len(v) == 0:
            continue

        result = False
        print(
            f"[repr error] The following variables were expected at breakpoint#{i}, but were not \
tested:\n  {v}"
        )

    for i, v in enumerate(tested_not_expected):
        if len(v) == 0:
            continue

        result = False
        print(
            f"[repr error] The following variables were tested, but do not exist in the input data \
at breakpoint#{i}:\n  {v}"
        )

    return result


def var_matches(var: Variable, expected: Variable, valobj: lldb.SBValue) -> Result:
    # Happy path requires very little intercession from us. We keep these values on the stack
    # so we don't have to recalculate them if we need to do error handling
    summary_ok = var.summary == expected.summary
    synthetic_ok = var.synthetic == expected.synthetic
    pretty_type_name_ok = var.pretty_type_name == expected.pretty_type_name
    pretty_print_ok = var.pretty_print == expected.pretty_print
    format_ok = var.format == expected.format

    type_ok = var.type == expected.type
    type_match_ok = type_matches(
        valobj.GetType(),
        valobj.GetTarget(),
        summary_ok & synthetic_ok & format_ok & pretty_type_name_ok & pretty_print_ok,
    )

    value_ok = var.value == expected.value

    work_list = [valobj.GetChildAtIndex(i) for i in range(valobj.GetNumChildren())]
    target = valobj.GetTarget()
    child_types_ok = True

    while len(work_list) != 0:
        obj = work_list.pop()

        for i in range(obj.GetNumChildren()):
            child = obj.GetChildAtIndex(i)
            # We don't need to report an error for invalid children here. Invalid objects can't be
            # blessed, thus should never exist in INPUT_DATA. That means they will always report
            # as a mismatch in `children_match`
            if child.IsValid():
                work_list.append(child)
            else:
                child_types_ok = False

        child_types_ok &= type_matches(obj.GetType(), target) == Result.Ok

    children_ok = children_match(
        var.children, expected.children, valobj.GetName(), valobj
    )

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

    error_source = f"var '{valobj.GetName()}'"

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
            error_source, "summary (Summary Provider)", var.summary, expected.summary
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
                provider = get_provider(var.summary)
                _ = provider(valobj, {})
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
            "synthetic (Synthetic Provider)",
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

        if not children_ok:
            # If the children don't match, we can check for more catastrophic failures using the
            # synthetic provider. All the per-children errors will have been printed in the
            # `children_match` check above.
            if var.synthetic is not None:
                try:
                    synth_provider = get_provider(var.synthetic)

                    # First we check for exceptions in the constructor and initialization
                    synth: lldb.SBSyntheticValueProvider = synth_provider(
                        valobj.GetNonSyntheticValue(), {}
                    )
                    synth.update()

                    # If the `get_child_at_index` function doesn't exist, there's not much more we
                    # can do
                    if getattr(synth, "get_child_at_index", None) is not None:
                        # If all the children are invalid (e.g. because a template arg isn't
                        # resolving correctly, incorrect enum discriminant), we should dump the
                        # internal state of the synthetic
                        if not all(
                            synth.get_child_at_index(i).IsValid()
                            for i in range(synth.num_children())
                        ):
                            dump_synthetic_state(synth)

                except Exception as e:
                    print_error(
                        error_source + " Synthetic",
                        "Error while running Synthetic\
Provider:",
                    )
                    traceback.print_exception(
                        type(e), e, e.__traceback__, file=sys.stdout
                    )

    return Result.Mismatch


def dump_synthetic_state(synth: Any):
    """Prints an object via builtin `vars()`. If `obj.__dict__` does not exist because the object is
    using `__slots__` intsead, the `__slots__` are converted into a dict and printed."""
    if (getattr(synth, "__dict__", None)) is not None:
        fields = vars(synth)
    elif (slots := getattr(synth, "__slots__", None)) is not None:
        fields = {name: getattr(synth, name, None) for name in slots}
    else:
        # Shouldn't be possible, but better safe than sorry
        print("Unable to print Synthetic Provider state")
        return

    print(f"Synthetic Provider state:\n  {fields})")


def children_match(
    children: list[Child],
    expected: list[Child],
    path: str,
    valobj: lldb.SBValue,
) -> Result:
    """Recursively checks children against an expected value and prints errors for mismatches."""

    result = Result.Ok if len(children) == len(expected) else Result.Mismatch

    mismatches = []
    missing = []
    invalid_count = 0

    for i in range(len(expected)):
        exp = expected[i]

        if i >= len(children):
            missing.append(exp.name)
            continue

        got = children[i]

        if got.name is None:
            result = Result.Mismatch
            invalid_count += 1
            mismatches.append(
                f"{exp.name}: {exp.type} = {exp.value} -> <Invalid SBValue>"
            )
        elif got.name != exp.name or got.type != exp.type or got.value != exp.value:
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
                valobj.GetChildAtIndex(i),
            )

    if result == Result.Ok:
        return result

    # If every single child is invalid, we can condense the output a lot by pointing to the
    # synthetic instead of printing a bunch of identical mismatches
    if invalid_count == len(children):
        print_error(
            path,
            f"All children of this object are invalid SBValue objects.\n    This is \
almost always caused by invalid state or logic in the SyntheticProvider.\n    This object's \
synthetic appears to be '{valobj.GetTypeSynthetic().GetData()}'",
        )
    elif len(mismatches) != 0:
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


def get_provider(provider_str: str) -> Callable[[lldb.SBValue, dict[Any, Any]], Any]:
    """Given a Varible.summary or Variable.Synthetic, imports the appropriate module and returns the
    matching Class/Function"""
    import importlib

    [module, summary_name] = provider_str.split(".", 1)
    provider_module = importlib.import_module(module)

    return getattr(provider_module, summary_name)
