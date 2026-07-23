"""Contains the class definitions outlining the schema of the test data. For LLDB conversion
from/into these types, see `./from_lldb.py`"""

import json
import os
from enum import Enum
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional, Union, get_origin, Final
from pprint import pformat

char = str
Primitive = Union[int, float, bool, char]
ByteSize = int

# see: default json decoder docs https://docs.python.org/3/library/json.html#json.JSONDecoder
# The types we're dealing with can only be: int, str, float, list, dict, bool, and None
JsonType = Union[int, str, float, list["JsonType"], bool, None, dict[str, "JsonType"]]


class Result(Enum):
    Ok = True
    Mismatch = False

    def __and__(self, other: "Result") -> "Result":
        return Result(self.value & other.value)

    def __bool__(self) -> bool:
        return self.value


ANSI_RED = "\033[91m"
ANSI_END = "\033[0m"


def print_error(error_source: str, message: str):
    print(f"{ANSI_RED}  [repr error: {error_source}]{ANSI_END} {message}")


def format_mismatch(label: str, got: Optional[Any], expected: Optional[Any]) -> str:
    if got is None and expected is not None:
        return f"{label} not found, expected: {expected}"
    elif expected is not None and got is None:
        return f"{label} '{got}' found when none was expected."
    else:
        return f"{label} does not match.\n    Expected: {expected}\n    Got: {got}"


def print_mismatch(
    error_source: str, label: str, got: Optional[Any], expected: Optional[Any]
):
    print_error(error_source, format_mismatch(label, got, expected))


class Target(Enum):
    """Due to the differences between PDB and DWARF debug info, we cannot guarantee their output
    will be identical. Since LLDB can handle both, we need to conditionally select the correct
    test data to use.

    Additionally, since there are differences in the internals of some structs based on OS (e.g.
    `PathBuf`/`OsString`), we need to be aware of whether we're on Windows or not.

    A global var `TARGET` is set to the current variant upon `lldb_test.py`'s instantiation using an
    env var passed from `compiletest` and is not expected to change afterwards."""

    NonWindows = "non_windows"
    WindowsGnu = "windows_gnu"
    WindowsMsvc = "windows_msvc"


def get_target() -> Target:
    # set by compiletest when launching LLDB
    t: str = os.environ["LLDB_BATCHMODE_TARGET_TRIPLE"]

    if t.endswith("windows-msvc"):
        return Target.WindowsMsvc
    if t.endswith("windows-gnu") or t.endswith("windows-gnullvm"):
        return Target.WindowsGnu

    return Target.NonWindows


BLESS: Final[bool] = os.environ["LLDB_BATCHMODE_BLESS_TEST_DATA"] == "1"
"""Global constant set by `compiletest` that determines whether or not we are blessing the test
data."""


TARGET: Final[Target] = get_target()
"""Global constant set by `compiletest`. Determines which target the tests were run for, thus which
set of test input we check."""


def annot_to_ty(annot: str) -> type[Any]:
    """Fallback to resolve a string type annotation to its actual type (e.g. `"Variable"` ->
    `Variable`). For types with generics, the generic is ignored."""

    return {
        "int": int,
        "float": float,
        "bool": bool,
        "None": type(None),
        "list": list,
        "dict": dict,
        "str": str,
        "ByteSize": int,
        "TargetData": TargetData,
        "Variable": Variable,
        "Type": Type,
        "Field": Field,
        "Child": Child,
        "BlessMetadata": BlessMetadata,
    }.get(annot.split("[", 1)[0], type[Any])


def from_dict(ty: type[Any], data: JsonType):
    """Translates a dictionary into an instance of the given dataclass type (with possibly nested
    dataclasses).

    Relies on accurate type hints for the dataclass's fields, and the default `dataclass.__init__`
    definition."""

    # Optional isn't a constructor, so we have to "unwrap" it.
    if get_origin(ty) is Optional:
        ty = ty.__args__[0]

    # recurse into lists
    if isinstance(data, list):
        # pulls the generic type from the list (e.g. `list[int]` -> `int`)
        inner = ty.__args__[0]
        if isinstance(inner, str):
            inner = annot_to_ty(inner)

        return [from_dict(inner, i) for i in data]

    if get_origin(ty) is dict and ty.__args__[0] is str:
        assert isinstance(data, dict)
        val_ty = ty.__args__[1]
        if isinstance(val_ty, str):
            val_ty = annot_to_ty(val_ty)

        if val_ty in [Variable, Child, Type, Field]:
            return {k: from_dict(val_ty, data[k]) for k in data.keys()}

    # map dict -> dataclass, recursing for each field
    if is_dataclass(ty):
        assert isinstance(data, dict)

        field_types = {f.name: f.type for f in fields(ty)}

        try:
            field_map = {}

            for f in data:
                f_type = field_types[f]

                # type annotations can be strings, so we need to resolve them to their actual type
                if isinstance(f_type, str):
                    f_type = annot_to_ty(f_type)

                field_map[f] = from_dict(f_type, data[f])

            # if you've never seen this before, `**` is the splat operator. It expands a mapping
            # type (in this case a dict) to keyword arguments. The ordering of the mapping does not
            # matter, only that the mapping's keys match the functions keyword args, and
            # `len(mapping)` == the number of keyword args.
            return ty(**field_map)
        except KeyError as e:
            print(
                f"Unable to convert dict to {ty}: Invalid field name {e}. If the test schema was \
changed intentionally, use the `--bless` option to update test data to the new schema."
            )

    # for any other type, we don't need to do any processing
    return data


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    """The fully qualified name of the field's type. Full type information should be looked up
    via `TargetData.types`"""

    offset: ByteSize


@dataclass
class Type:
    size: ByteSize
    # When GDB support is added to the test framework, basic_type and type_class will probably be
    # converted to a wrapper IntEnum that converts GDB's equivalent information to
    basic_type: int
    """The `lldb.eBasicType` value associated with this type. Tested due to our use of it in type
    recognizer functions."""

    type_class: int
    """The `lldb.eTypeClass` value associated with thjs type. Tested due to our use of it in type
    recognizer functions."""

    fields: list[Field]
    """Stored as a list due to our reliance on `SBType.GetFieldAtIndex()`

    Note: LLDB **does not** reorder the fields of a type based on their offset. For example,
    `GetFieldAtIndex(0).GetByteOffset()` may return `8`. Instead, the order of the fields is a
    direct reflection of their ordering in the debug info (which, as far as I know, is the same as
    their declaration order in the source code).
    """

    generic_params: list[str]
    """Stored as a list due to our reliance on `SBType.GetTemplateArgumentType()` and the sequential
    behavior of `lldb_providers.get_template_args`"""
    # FIXME the only way we can look up static fields is by name (as of lldb 22), so we need a way
    # to discover them. ATM only sum-type enums on MSVC use static fields, and those are fixed
    # values, so it's not super urgent.
    # static_fields: list[StaticField]

    def matches(
        self, expected: "Type", type_name: str, provider_ok: bool = False
    ) -> Result:
        result = Result.Ok
        error_source = f"type '{type_name}'"
        # FIXME handle 32 bit targets
        if self.size != expected.size:
            result = Result.Mismatch
            print_mismatch(error_source, "size", self.size, expected.size)

        if self.fields != expected.fields:
            result = Result.Mismatch
            self.print_field_errors(expected, error_source)

        if self.generic_params != expected.generic_params:
            result = Result.Mismatch
            print_mismatch(
                error_source,
                "generic_params",
                self.generic_params,
                expected.generic_params,
            )

        if result == Result.Mismatch and provider_ok:
            print_error(
                error_source,
                "It appears these changes do not affect the type's providers. Consider rerunning \
with the `--bless` option",
            )

        return result

    def print_field_errors(self, expected: "Type", error_source: str):
        """Extra processing for better error messages. The following common cases are covered:
        * New/Missing fields
        * Source code rearranged fields
        * Rustc rearranged fields
        * Renamed fields

        If none of the common cases are encountered, we just generically print any mismatched
        fields.
        """

        # FIXME these checks aren't exactly the most efficient they could be. Luckily, the happy
        # path skips this function entirely, so passing tests are still fast. These checks could
        # probably all be done in 2ish total iters over each list, but optimization isn't a huge
        # concern at the moment.

        got_set = set(self.fields)
        expected_set = set(expected.fields)

        if len(self.fields) != len(expected.fields):
            new_fields = got_set.difference(expected_set)

            missing_fields = expected_set.difference(got_set)

            if len(missing_fields) != 0:
                print_error(
                    error_source,
                    f"The following field(s) appear to have been removed from the type:\n\
{missing_fields}",
                )

            if len(new_fields) != 0:
                print_error(
                    error_source,
                    f"The following field(s) appear to have been added to the type:\n\
{new_fields}",
                )

        # are all of the same fields present, regardless of order? If so, they were rearranged
        # in the source code, but the compiler kept the same ordering.
        elif got_set == expected_set:
            print_error(
                error_source,
                f"Field(s) appear to have been rearranged:\n    Expected:\n\
{pformat(self.fields, indent=6)}\n    Got:\n{pformat(expected.fields, indent=6)}",
            )
        else:
            # we know for sure that
            types_match = True
            offsets_match = True
            names_match = True
            mismatches: list[tuple[Field, Field]] = []

            for g, e in zip(self.fields, expected.fields):
                if g.type != e.type:
                    types_match = False
                    mismatches.append((g, e))
                if g.offset != e.offset:
                    offsets_match = False
                    mismatches.append((g, e))
                if g.name != e.name:
                    names_match = False
                    mismatches.append((g, e))

            # If the types and offsets are the same but the names aren't, we know fields have
            # been renamed.
            if types_match and offsets_match:
                renames = "\n    ".join(
                    map(lambda m: f"{m[1].name} -> {m[0].name}", mismatches)
                )
                print_error(
                    error_source,
                    f"The following field(s) appear to have been renamed (expected -> got):\n\
    {renames}",
                )

            # If the types and names are the same, but the offsets are different, we know that rustc
            # has decided to order the fields differently, despite the source code not changing
            elif types_match and names_match:
                reordered = "\n    ".join(
                    map(
                        lambda m: (
                            f"{m[1].name} offset: +{m[1].offset} -> {m[0].name} offset: \n\
+{m[0].offset}"
                        ),
                        mismatches,
                    )
                )

                print_error(
                    error_source,
                    f"The following field(s) appear to have been reordered by rustc (expected -> \
got):\n    {reordered}",
                )

            else:
                mm_string = "\n    ".join(
                    map(
                        lambda m: (f"{m[1]} -> {m[0]}"),
                        mismatches,
                    )
                )

                print_error(
                    error_source,
                    f"The following field(s) do not match (expected -> got):\n\
    {mm_string}",
                )


@dataclass
class Child:
    """Similar to `Variable`, but carries less information since we primarily test top-level
    values (and assume values of these child types have been tested thoroughly elsewhere).

    Note that if the type has a synthetic provider (lldb) or pretty printer (gdb), the child names
    and types can be set to anything at all, so we do need to test these separately from the
    parent's type's fields."""

    name: str
    """The name used to access the child. If the parent object has a synthetic, the child name can
    be overridden."""

    type: str
    """The fully qualified name of the child's type. Full type information should be looked up
    via `TargetData.types`"""

    value: Optional[Primitive]
    children: list["Child"]
    """Children are stored as a list because of our use of `GetChildAtIndex()`. Providers can also
    dictate the order that children populate, so it's important to ensure that stays consistent too.
    """


@dataclass
class Variable:
    type: str
    """The fully qualified name of the variable's type. Full type information should be looked up
    via `TargetData.types`"""

    pretty_type_name: Optional[str]
    """Type names can be overridden by `SyntehticProvider.get_type_name()` in LLDB and by
    `type_printer` in GDB"""

    pretty_print: Optional[str]
    """The string-result of pretty printing the value (`SBValue.GetSummary` for LLDB,
    `pretty_printer.to_string` for GDB). `None` for aggregates with no summary provider."""

    value: Optional[Primitive]
    """`None` if the object does not have a primitive representation."""

    synthetic: Optional[str]
    """The class/function name of the synthetic provider (lldb) or pretty printer (gdb).
    `None` if the object does not have a synthetic provider"""

    summary: Optional[str]
    """The function name of the summary provider. `None` if the object does not have a summary
    provider, or if the test data is for GDB"""

    format: Optional[int]
    """The `lldb.eFormat` enum variant associated with this type (if applicable)."""

    # Stored as a list instead of a dict because child order matters
    children: list[Child]
    """A list of children provided by the object. If the object has a synthetic provider, the
    children are the result of the provider's `get_child_at_index` function"""


@dataclass
class BlessMetadata:
    """
    Contains additional context about the tools at the time the test data was generated.
    """

    python_version: str = ""
    debugger_version: str = ""
    feature_flags: str = ""


@dataclass
class TargetData:
    """
    Top-level container for all test data.

    Due to the differences between PDB and DWARF debug info, we cannot guarantee their output
    will be identical. Since LLDB can handle both, we need to conditionally select the correct
    test data to use.

    Additionally, since there are differences in the internals of some structs based on OS (e.g.
    `PathBuf`/`OsString`), we need to be aware of whether we're on Windows or not.

    A global var `TARGET` is set to the current variant upon `lldb_batchmode`'s instantiation using
    an env var passed from `compiletest` and is not expected to change afterwards.
    """

    bless_metadata: BlessMetadata = field(default_factory=BlessMetadata)
    """Miscellaneous data included to make diagnosing issues easier. This data is not intended to be
    tested against."""

    types: dict[str, Type] = field(default_factory=dict)
    """
    A map of type names to types. Contains all types present in the test's variables, including the
    types of fields and child objects.
    """

    # If we ever decide that it makes sense to check the same variable twice at the same breakpoint
    # this will need to be converted to a list
    breakpoints: list[dict[str, Variable]] = field(default_factory=list)
    """Each element corresponds to one stopping point in the test. The element itself is a
    dictionary mapping variable names to their respective test data."""

    @staticmethod
    def initialize() -> "TargetData":
        result = TargetData()
        path = os.environ["LLDB_BATCHMODE_INPUT_DATA_PATH"]
        if not os.path.isfile(path):
            if BLESS:
                return result
            else:
                raise Exception(
                    f"Invalid input data path: '{path}'\nIf test data has not been \
generated for this test yet, consider using the `--bless` option."
                )

        if BLESS:
            return result

        with open(path, "r") as f:
            try:
                result = from_dict(TargetData, json.load(f))
            except json.decoder.JSONDecodeError:
                print("Warning: Malformed input data, reverting to default")

        return result

    def save_blessing(self, metadata: BlessMetadata):
        """Writes the entirety of `self` to the env var `LLDB_BATCHMODE_INPUT_DATA_PATH`, which is
        set by `compiletest` before running `lldb_batchmode. Used to finalize changes made by one or
        more `from_lldb.bless_variable` calls.

        This function should be called exactly once, right before
        `lldb_batchmode.runner.main` exits if the following conditions are met:

        1. No other exceptions or error states occurred
        2. `BLESS == True`
        3. At least one `repr` pseudo-command was processed

        This prevents us from saving incomplete data or invalid data. It also prevents us from
        creating input data files for tests that do not need it.
        """

        self.bless_metadata = metadata
        path = os.environ["LLDB_BATCHMODE_INPUT_DATA_PATH"]
        # dumping directly to a file is somewhat unsafe. If the `Variable`/`Type` data ends up in a
        # state that cannot be serialized correctly, the json ends up malformed, and we could end up
        # overwriting valid test data with a complete mess. Since the in-memory data is typically
        # completely valid, the testing logic will pass and make it seem like nothing is wrong.

        # While we could rely on git to help revert the test file, it's better to just not allow it
        # to save malformed json in the first place. Thus, we dump the JSON, re-read it to check
        # for `JSONDecodeError`, and write it to the target file if no error occurred.
        x = json.dumps(asdict(self), indent=" ")
        _ = json.loads(x)

        # ensure the necessary directories exist first
        import pathlib

        os.makedirs(pathlib.Path(path).parent, exist_ok=True)

        with open(path, "w") as f:
            f.write(x)


INPUT_DATA: TargetData = TargetData.initialize()
