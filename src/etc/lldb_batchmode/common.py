# Contains the class definitions outlining the schema of the test data

import enum
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import NoneType
from typing import Any, Optional, get_origin, Final

char = str
Primitive = int | float | bool | char
ByteSize = int

# see: default json decoder docs https://docs.python.org/3/library/json.html#json.JSONDecoder
# The types we're dealing with can only be: int, str, float, list, dict, bool, and None
JsonType = int | str | float | list["JsonType"] | bool | None | dict[str, "JsonType"]


class Target(enum.Enum):
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
        "None": NoneType,
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

    Relies on accurate type hints for the dataclass's fields, and the standard `dataclass.__init__`
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


@dataclass(slots=True)
class Field:
    name: str
    type: str
    """The fully qualified name of the field's type. Full type information should be looked up
    via `TargetData.types`"""

    offset: ByteSize


@dataclass(slots=True)
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
    """Stored as a list due to our reliance on `SBType.GetFieldAtIndex()`"""

    generic_params: list[str]
    """Stored as a list due to our reliance on `SBType.GetTemplateArgumentType()` and the sequential
    behavior of `lldb_providers.get_template_args`"""
    # FIXME the only way we can look up static fields is by name (as of lldb 22), so we need a way
    # to discover them. ATM only sum-type enums on MSVC use static fields, and those are fixed
    # values, so it's not super urgent.
    # static_fields: list[StaticField]


@dataclass(slots=True)
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


@dataclass(slots=True)
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


@dataclass(slots=True)
class BlessMetadata:
    """
    Contains additional context about the tools at the time the test data was generated
    """

    python_version: str = ""
    debugger_version: str = ""
    # FIXME (todo)
    # feature_flags: str


@dataclass(slots=True)
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

        with open(path, "r") as f:
            try:
                result = from_dict(TargetData, json.load(f))
            except json.decoder.JSONDecodeError:
                print("Warning: Malformed input data, reverting to default")

        return result

    def save_blessing(self, metadata: BlessMetadata):
        """Writes the entirety of `self` to the input file. Used to finalize changes made by
        one or more `TestData.bless_variable` calls."""

        self.bless_metadata = metadata
        path = os.environ["LLDB_BATCHMODE_INPUT_DATA_PATH"]
        # dumping directly to a file is somewhat unsafe. If the json ends up malformed, we could
        # end up overwriting valid test data with a complete mess. Since the in-memory data
        # typically *isn't* malformed, the `--bless` will pass and make it seem like nothing is
        # wrong.

        # While we could rely on git to help revert the test file, it's better to just not allow it
        # to save malformed json in the first place. Thus, we dump the JSON, re-read it, and then
        # only when that succeeds do we save it.
        x = json.dumps(asdict(self), indent=" ")
        _ = json.loads(x)

        with open(path, "w") as f:
            f.write(x)
