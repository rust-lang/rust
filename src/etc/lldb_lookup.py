from __future__ import annotations
from typing import TYPE_CHECKING, List, Callable

import lldb

from lldb_providers import (
    LLDBVersion,
    DBG_VERSION,
    LLDBOpaque,
    StructSyntheticProvider,
    StdHashMapSyntheticProvider,
    StdOldHashMapSyntheticProvider,
    StdRcSyntheticProvider,
    TupleSyntheticProvider,
    EmptySyntheticProvider,
    ClangEncodedEnumProvider,
    DefaultSyntheticProvider,
    StdStringSyntheticProvider,
    StdStringSummaryProvider,
    StdSliceSyntheticProvider,
    StdStrSummaryProvider,
    MSVCStrSyntheticProvider,
    SizeSummaryProvider,
    MSVCStdSliceSyntheticProvider,
    StdSliceSummaryProvider,
    StdOsStringSummaryProvider,
    StdVecSyntheticProvider,
    StdVecDequeSyntheticProvider,
    StdRcSummaryProvider,
    StdCellSyntheticProvider,
    StdRefSyntheticProvider,
    StdRefSummaryProvider,
    StdNonZeroNumberSummaryProvider,
    StdPathBufSummaryProvider,
    MSVCEnumSyntheticProvider,
    MSVCEnumSummaryProvider,
    TupleSummaryProvider,
    MSVCTupleSyntheticProvider,
    ClangEncodedEnumSummaryProvider,
)
from rust_types import (
    ENUM_DISR_FIELD_NAME,
    ENUM_LLDB_ENCODED_VARIANTS,
    RustType,
    classify_union,
    is_tuple_fields,
)

if TYPE_CHECKING:
    from lldb import SBValue

####################################################################################################
# This file contains lookup functions that associate rust types with their synthetic/summary
# providers.
#
# LLDB caches the results of the the commands in `lldb_commands`, but that caching is "shallow". It
# purely associates the type with the function given, whether it is a regular function or a class
# constructor. If the function makes decisions about what type of SyntheticProvider to return, that
# processing is done **each time a value of that type is encountered**.
#
# To reiterate, inspecting a `vec![T; 100_000]` will call `T`'s lookup function/constructor 100,000
# times. This can lead to significant delays in value visualization if the lookup logic is complex.
#
# As such, lookup functions should be kept as minimal as possible. LLDB technically expects a
# SyntheticProvider class constructor. If you can provide just a class constructor, that should be
# preferred. If extra processing must be done, try to keep it as minimal and as targeted as possible
# (see: `classify_hashmap()` vs `classify_hashset()`).
####################################################################################################

RUST_CATEGORY: lldb.SBTypeCategory = lldb.SBTypeCategory()
MOD_PREFIX = f"{__name__}."

DEFAULT_TYPE_OPTIONS: int = (
    # ensure it applies through typedef chains
    lldb.eTypeOptionCascade
    | lldb.eTypeOptionHideEmptyAggregates
    # helps us reason about what types can be put through the synthetic provider.
    # this is important because of the `update` logic as well as when working with
    # type information, since we almost always want to work on the pointee type/pointee's
    # template args. These options allow us to know that the type we have is never a pointer-to
    # the type we want
    # | lldb.eTypeOptionSkipPointers
    # | lldb.eTypeOptionSkipReferences
    | lldb.eTypeOptionFrontEndWantsDereference
)


def __lldb_init_module(debugger: lldb.SBDebugger, _dict: LLDBOpaque):
    global RUST_CATEGORY
    RUST_CATEGORY = debugger.GetCategory("Rust")

    if not RUST_CATEGORY.IsValid():
        RUST_CATEGORY = debugger.CreateCategory("Rust")

    RUST_CATEGORY.SetEnabled(True)

    # BUG: This specifier is used to determine whether or not visualizers in this category shoud be
    # used for the given executable. It defaults to `lldb.eLanguageTypeUnknown` which will be active
    # regardless of the executable's reported language, but setting it to `rust` would be ideal
    # so that our visualizers do not apply to, for example, C++ code when debugging across FFI
    # boundaries.
    # We cannot currently enable this though, as this query defers to the `TypeSystem` (in our case,
    # `TypeSystemClang`). `TypeSystemClang::SupportsLanguage` includes Rust, which allows Rust
    # support in LLDB, but SBTypeCategory instead checks
    # `TypeSystemClang::GetSupportedLanguagesForTypes` which only includes various C and C++
    # versions. I'm not sure if Rust should be added to that function or not, but in the meantime
    # cannot directly specify that Rust is the intended language for this category.

    # RUST_CATEGORY.AddLanguage(lldb.eLanguageTypeRust)

    version_str: str = debugger.version

    if version_str.startswith("lldb version "):
        # e.g. "lldb version 22.1.2 (https://github.com/llvm/llvm-project revision
        # 1ab49a973e210e97d61e5 db6557180dcb92c3e98)\n  clang revision
        # 1ab49a973e210e97d61e5db6557180dcb92c3e98\n  llvm revision
        # 1ab49a973e210e97d61e5db6557180dcb92c3e98"

        # this sould lop off the front and back, leaving us with only the 'x.x.x'
        version_str: str = version_str.removeprefix("lldb version ").split(" ", 1)[0]

        major, minor, patch = version_str.split(".", 3)

        if not patch.isdigit():
            i = 0
            while i < len(patch) and patch[i].isdigit():
                i += 1
            patch = patch[:i]
    elif version_str.startswith("lldb-"):
        # FIXME: apple uses custom lldb. Their versioning seems to not directly correspond to LLVM's
        # at some point it'll be worth adding in explicit tests for features rather than version
        # checks.
        major, minor, patch = 100, 0, 0

    global DBG_VERSION
    DBG_VERSION = LLDBVersion(int(major), int(minor), int(patch))

    register_providers_compatibility()


def register_providers_compatibility():
    """
    Adds providers to global `RUST_CATEGORY`. Does not attempt to use type recognizers

    The order that providers are added matters. Existing providers are iterated through in **reverse
    order** when finding a match, allowing new providers to "overwrite" old providers. Be very
    careful when modifying the order that providers are added.
    """

    global RUST_CATEGORY

    if DBG_VERSION.has_type_recognizers():
        # FIXME: this can be removed once full support for type recognizers is added.
        # This prevents a semi-unfixable regression for CodeLLDB
        register_synth(
            synthetic_lookup,
            lldb.SBTypeNameSpecifier(
                MOD_PREFIX + not_primitive.__name__,
                lldb.eFormatterMatchCallback,
            ),
            lldb.eTypeOptionCascade,
        )
    else:
        # Need to toss any remaining types through this so that GNU enums are caught
        register_synth(
            synthetic_lookup,
            lldb.SBTypeNameSpecifier(r".*", True),
        )

    # String
    register(
        StdStringSyntheticProvider,
        StdStringSummaryProvider,
        r"^(alloc::([a-z_]+::)+)String$",
    )

    # str GNU
    register(
        StdSliceSyntheticProvider,
        StdStrSummaryProvider,
        r"^&(mut )?str$",
    )

    # str MSVC
    register(
        MSVCStrSyntheticProvider,
        StdStrSummaryProvider,
        r"^ref(_mut)?\$<str\$>$",
    )

    # slice GNU
    register(
        StdSliceSyntheticProvider,
        SizeSummaryProvider,
        r"^&(mut )?\[.+\]$",
    )

    # slice MSVC
    register(
        MSVCStdSliceSyntheticProvider,
        StdSliceSummaryProvider,
        r"^ref(_mut)?\$<slice2\$<.+> >",
    )

    # OsString
    register_summary(
        StdOsStringSummaryProvider,
        lldb.SBTypeNameSpecifier(r"^(std::ffi::([a-z_]+::)+)OsString$", True),
    )

    # Vec
    register(
        StdVecSyntheticProvider,
        SizeSummaryProvider,
        r"^(alloc::([a-z_]+::)+)Vec<.+>$",
    )

    # VecDeque
    register(
        StdVecDequeSyntheticProvider,
        SizeSummaryProvider,
        r"^(alloc::([a-z_]+::)+)VecDeque<.+>$",
    )

    # HashMap
    register(
        classify_hashmap,
        SizeSummaryProvider,
        r"^(std::collections::([a-z_]+::)+)HashMap<.+>$",
    )

    # HashSet
    register(
        classify_hashset,
        SizeSummaryProvider,
        r"^(std::collections::([a-z_]+::)+)HashSet<.+>$",
    )

    # Rc
    register(
        StdRcSyntheticProvider,
        StdRcSummaryProvider,
        r"^(alloc::([a-z_]+::)+)Rc<.+>$",
    )

    # Arc
    register(
        arc_synthetic,
        StdRcSummaryProvider,
        r"^(alloc::([a-z_]+::)+)Arc<.+>$",
    )

    # Cell
    register_synth(
        StdCellSyntheticProvider,
        lldb.SBTypeNameSpecifier(
            r"^(core::([a-z_]+::)+)Cell<.+>$",
            True,
        ),
    )

    # RefCell
    register(
        StdRefSyntheticProvider,
        StdRefSummaryProvider,
        r"^(core::([a-z_]+::)+)Ref(Cell|Mut)?<.+>$",
    )

    # NonZero
    register_summary(
        StdNonZeroNumberSummaryProvider,
        lldb.SBTypeNameSpecifier(
            r"^(core::([a-z_]+::)+)NonZero(<.+>|I\d{0,3}|U\d{0,3})$",
            True,
        ),
    )

    # PathBuf
    register_summary(
        StdPathBufSummaryProvider,
        lldb.SBTypeNameSpecifier(
            r"^(std::([a-z_]+::)+)PathBuf$",
            True,
        ),
    )

    # Path
    register_summary(
        StdNonZeroNumberSummaryProvider,
        lldb.SBTypeNameSpecifier(
            r"^&(mut )?(std::([a-z_]+::)+)Path$",
            True,
        ),
    )

    # Enum MSVC
    register(
        MSVCEnumSyntheticProvider,
        MSVCEnumSummaryProvider,
        r"^enum2\$<.+>$",
    )

    # Tuple GNU
    register(
        TupleSyntheticProvider,
        TupleSummaryProvider,
        r"^\(.*\)$",
        type_options=DEFAULT_TYPE_OPTIONS | lldb.eTypeOptionHideChildren,
    )

    # Tuple MSVC
    register(
        MSVCTupleSyntheticProvider,
        TupleSummaryProvider,
        r"^tuple\$<.+>$",
    )


def register(
    synth_provider: Callable[[SBValue, LLDBOpaque], object],
    summary_provider: Callable[[SBValue, LLDBOpaque], str],
    match_str,
    regex: bool = True,
    type_options: int = DEFAULT_TYPE_OPTIONS,
):
    global RUST_CATEGORY
    sb_name: lldb.SBTypeNameSpecifier = lldb.SBTypeNameSpecifier(match_str, regex)

    register_synth(synth_provider, sb_name, type_options)
    register_summary(summary_provider, sb_name, type_options)


def register_synth(
    provider: Callable[[SBValue, LLDBOpaque], object],
    sb_name: lldb.SBTypeNameSpecifier,
    type_options: int = DEFAULT_TYPE_OPTIONS,
):
    sb_synth: lldb.SBTypeSynthetic = lldb.SBTypeSynthetic.CreateWithClassName(
        MOD_PREFIX + provider.__name__
    )
    sb_synth.SetOptions(type_options)

    global RUST_CATEGORY
    # returns false for failures, does not provide any info to determine what the failure was

    res: bool = RUST_CATEGORY.AddTypeSynthetic(sb_name, sb_synth)

    if not res:
        print(
            "Warning: unable to register summary: "
            + f"{MOD_PREFIX + provider.__name__} with specifier '{sb_name.GetName()}'"
        )


def register_summary(
    provider: Callable[[SBValue, LLDBOpaque], object],
    sb_name: lldb.SBTypeNameSpecifier,
    type_options: int = DEFAULT_TYPE_OPTIONS,
):
    sb_summary: lldb.SBTypeSummary = lldb.SBTypeSummary.CreateWithFunctionName(
        MOD_PREFIX + provider.__name__
    )

    sb_summary.SetOptions(type_options)

    global RUST_CATEGORY
    # returns false for failures, does not provide any info to determine what the failure was
    res: bool = RUST_CATEGORY.AddTypeSummary(sb_name, sb_summary)

    if not res:
        print(
            "Warning: unable to register summary: "
            + f"{MOD_PREFIX + provider.__name__} with specifier '{sb_name.GetName()}'"
        )


# BACKCOMPAT: rust 1.35
def is_hashbrown_hashmap(hash_map: lldb.SBValue) -> bool:
    return len(hash_map.type.fields) == 1


def classify_hashmap(valobj: lldb.SBValue, _dict: LLDBOpaque) -> object:
    if is_hashbrown_hashmap(valobj):
        return StdHashMapSyntheticProvider(valobj, _dict)
    else:
        return StdOldHashMapSyntheticProvider(valobj, _dict)


def classify_hashset(valobj: lldb.SBValue, _dict: LLDBOpaque) -> object:
    hash_map = valobj.GetChildAtIndex(0)
    if is_hashbrown_hashmap(hash_map):
        return StdHashMapSyntheticProvider(valobj, _dict, show_values=False)
    else:
        return StdOldHashMapSyntheticProvider(hash_map, _dict, show_values=False)


def arc_synthetic(valobj: lldb.SBValue, _dict: LLDBOpaque) -> object:
    return StdRcSyntheticProvider(valobj, _dict, is_atomic=True)


def not_primitive(type: lldb.SBType, _dict: LLDBOpaque) -> bool:
    return type.GetBasicType() == lldb.eBasicTypeInvalid


def classify_rust_type(type: lldb.SBType, is_msvc: bool) -> RustType:
    if type.IsPointerType():
        return RustType.Indirection

    # there is a bit of code duplication here because we don't want to check all of the standard
    # library regexes since LLDB handles that for us
    type_class = type.GetTypeClass()
    if type_class == lldb.eTypeClassStruct:
        fields: List[lldb.SBTypeMember] = type.fields
        if len(fields) == 0:
            return RustType.Empty

        # <<variant>> is emitted by GDB while LLDB(18.1+) emits "$variants$"
        if (
            fields[0].name == ENUM_DISR_FIELD_NAME
            or fields[0].name == ENUM_LLDB_ENCODED_VARIANTS
        ):
            return RustType.Enum

        if is_tuple_fields(fields):
            return RustType.Tuple

        return RustType.Struct
    if type_class == lldb.eTypeClassUnion:
        # If we're debugging msvc, sum-type enums should have been caught by the regex in lldb
        # commands since they all start with "enum2$<"
        if is_msvc:
            return RustType.Union
        return classify_union(type.fields)

    return RustType.Other


def synthetic_lookup(valobj: lldb.SBValue, _dict: LLDBOpaque) -> object:
    """Returns the synthetic provider for the given value"""

    # small hack to check for the DWARF debug info section, since SBTarget.triple and
    # SBProcess.triple report lldb's target rather than the executable's. SBProcessInfo.triple
    # returns a triple without the ABI. It is also possible for any of those functions to return a
    # None object.
    # Instead, we look for the GNU `.debug_info` section, as MSVC does not have one with the same
    # name
    # FIXME: I don't know if this works when the DWARF lives in a separate file
    # (see: https://gcc.gnu.org/wiki/DebugFissionDWP). Splitting the DWARF is very uncommon afaik so
    # it should be okay for the time being.
    is_msvc = not valobj.GetFrame().GetModule().FindSection(".debug_info").IsValid()

    rust_type = classify_rust_type(valobj.GetType(), is_msvc)

    if rust_type == RustType.Struct or rust_type == RustType.Union:
        return StructSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StructVariant:
        return StructSyntheticProvider(valobj, _dict, is_variant=True)
    if rust_type == RustType.Tuple:
        return TupleSyntheticProvider(valobj, _dict)
    if rust_type == RustType.TupleVariant:
        return TupleSyntheticProvider(valobj, _dict, is_variant=True)
    if rust_type == RustType.Empty:
        return EmptySyntheticProvider(valobj, _dict)
    if rust_type == RustType.RegularEnum:
        discriminant = valobj.GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned()
        return synthetic_lookup(valobj.GetChildAtIndex(discriminant), _dict)
    if rust_type == RustType.SingletonEnum:
        return synthetic_lookup(valobj.GetChildAtIndex(0), _dict)
    if rust_type == RustType.Enum:
        # this little trick lets us treat `synthetic_lookup` as a "recognizer function" for the enum
        # summary providers, reducing the number of lookups we have to do. This is a huge time save
        # because there's no way (via type name) to recognize sum-type enums on `*-gnu` targets. The
        # alternative would be to shove every single type through `summary_lookup`, which is
        # incredibly wasteful. Once these scripts are updated for LLDB 19.0 and we can use
        # `--recognizer-function`, this hack will only be needed for backwards compatibility.
        summary: lldb.SBTypeSummary = valobj.GetTypeSummary()
        if (
            summary.summary_data is None
            or summary.summary_data.strip()
            != "lldb_lookup.ClangEncodedEnumSummaryProvider(valobj,internal_dict)"
        ):
            global RUST_CATEOGRY
            RUST_CATEGORY.AddTypeSummary(
                lldb.SBTypeNameSpecifier(valobj.GetTypeName()),
                lldb.SBTypeSummary().CreateWithFunctionName(
                    MOD_PREFIX + ClangEncodedEnumSummaryProvider.__name__
                ),
            )

        return ClangEncodedEnumProvider(valobj, _dict)
    # if rust_type == RustType.Indirection:
    #     return IndirectionSyntheticProvider(valobj, _dict)

    return DefaultSyntheticProvider(valobj, _dict)
