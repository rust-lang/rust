from __future__ import annotations
from typing import List


import lldb

from lldb_providers import *
from rust_types import (
    ENUM_DISR_FIELD_NAME,
    ENUM_LLDB_ENCODED_VARIANTS,
    RustType,
    classify_union,
    is_tuple_fields,
)

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
            rust_category: lldb.SBTypeCategory = lldb.debugger.GetCategory("Rust")
            rust_category.AddTypeSummary(
                lldb.SBTypeNameSpecifier(valobj.GetTypeName()),
                lldb.SBTypeSummary().CreateWithFunctionName(
                    "lldb_lookup.ClangEncodedEnumSummaryProvider"
                ),
            )

        return ClangEncodedEnumProvider(valobj, _dict)
    if rust_type == RustType.Indirection:
        return IndirectionSyntheticProvider(valobj, _dict)

    return DefaultSyntheticProvider(valobj, _dict)
