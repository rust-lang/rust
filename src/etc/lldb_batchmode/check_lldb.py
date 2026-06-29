"""Contains the logic that compares variables to `INPUT_DATA` via the entrypoint
`check(var_name, breakpoint_idx, frame)`. These comparisons report errors to stdout, and then return
a `Result` indicating whether or not the variable matched.

Checks *do not* stop after the first encountered error. Some redundant information may be ommitted
(e.g. checking pretty printed type name if the synthetic isn't properly attached to the type).
"""

from typing import Any

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
from .from_lldb import bless_variable, variable_from_lldb, type_from_lldb, get_generics


VARS_TESTED: list[dict[str, Result]] = []
"""Used to help ensure all expected variables were tested. Each element of the list corresponds to a
breakpoint, and contains a set of all of the variable names tested for that breakpoint."""


def check(var_name: str, breakpoint_idx: int, frame: lldb.SBFrame) -> Result:
    """`lldb-repr` pseudo-command entrypoint. Checks the variable against `INPUT_DATA` for the given
    frame at the given breakpoint. Returns True if a
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

    expected = INPUT_DATA.breakpoints[breakpoint_idx][var_name]

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


def type_matches(sbtype: lldb.SBType, sbtarget: lldb.SBTarget) -> Result:
    """Checks a type and all field/generic types (recursively) against the data contained in
    `INPUT_DATA`."""
    name: str = sbtype.GetName()
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
        # FIXME improve logic
        result = ty.matches(expected, name)

    TYPES_TESTED[name] = result

    fields: list[lldb.SBTypeMember] = sbtype.fields
    inner_types = [f.GetType() for f in fields]
    inner_types.extend(get_generics(sbtype, sbtarget))

    for t in inner_types:
        result = result and type_matches(t, sbtarget)

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

    type_ok = type_matches(valobj.GetType(), valobj.GetTarget())

    pretty_type_name_ok = var.pretty_type_name == expected.pretty_type_name
    pretty_print_ok = var.pretty_print == expected.pretty_print
    value_ok = var.value == expected.value
    format_ok = var.format == expected.format

    child_types_ok = Result.Ok

    target = valobj.GetTarget()

    work_list = [valobj]
    while len(work_list) != 0:
        obj = work_list.pop()
        work_list.extend([obj.GetChildAtIndex(i) for i in range(obj.GetNumChildren())])
        child_types_ok &= type_matches(obj.GetType(), target)

    children_ok = children_match(
        var.children, expected.children, valobj.GetName(), valobj
    )

    if (
        type_ok
        and pretty_type_name_ok
        and pretty_print_ok
        and value_ok
        and synthetic_ok
        and summary_ok
        and format_ok
        and children_ok
    ):
        return Result.Ok & child_types_ok

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
            import lldb_providers

            got_children = set(var.children)
            expected_children = set(expected.children)

            # do any children match at all?
            if len(got_children.difference(expected_children)) != 0:
                # FIXME (todo) similar checks to `Type.matches`
                pass
            # If none of the children match, we can check for more catastrophic failures using
            # the synthetic provider
            elif (
                var.synthetic is not None
                and (
                    synth_provider := getattr(
                        lldb_providers,
                        var.synthetic,
                        None,
                    )
                )
                is not None
            ):
                try:
                    synth: lldb.SBSyntheticValueProvider = synth_provider(
                        valobj.GetNonSyntheticValue(), {}
                    )
                    synth.update()

                    # If the children are invalid (e.g. because a template arg isn't resolving
                    # correctly), we should dump the internal state of the synthetic
                    if not all(
                        synth.get_child_at_index(i).IsValid()
                        for i in range(synth.num_children())
                    ):
                        dump_synthetic_state(synth)

                except Exception:
                    # FIXME (todo) print stack trace and error message that provider is failing
                    pass

            else:
                # If none of the children match and there's no synthetic, just print the regular
                # diff
                children_match(
                    var.children, expected.children, valobj.GetName(), valobj
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
    children: list[Child], expected: list[Child], path: str, valobj: lldb.SBValue
) -> Result:
    """Recursively checks children against an expected value and prints errors for mismatches."""

    result = Result.Ok

    for i, (got, exp) in enumerate(zip(children, expected)):
        # handle top level first, then we can recurse into the children
        got_fields = {name: getattr(got, name, None) for name in got.__slots__}
        got_children = got_fields.pop("children")

        exp_fields = {name: getattr(exp, name, None) for name in exp.__slots__}
        exp_children = exp_fields.pop("children")

        if got_fields != exp_fields:
            result = Result.Mismatch
            print_mismatch(
                path, f"{path}.{got.name} (Child #{i})", got_fields, exp_fields
            )

        if got_children is not None or exp_children is not None:
            result &= children_match(
                got_children,
                exp_children,
                f"{path}.{got.name}",
                valobj.GetChildAtIndex(i),
            )

    return result
