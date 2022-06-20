import lldb

from lldb_providers import *
from rust_types import RustType, classify_struct, classify_union


# BACKCOMPAT: rust 1.35
def is_hashbrown_hashmap(hash_map):
    return len(hash_map.type.fields) == 1


def classify_rust_type(type):
    type_class = type.GetTypeClass()
    if type_class == lldb.eTypeClassStruct:
        return classify_struct(type.name, type.fields)
    if type_class == lldb.eTypeClassUnion:
        return classify_union(type.fields)

    return RustType.OTHER


def summary_lookup(valobj, dict):
    # type: (SBValue, dict) -> str
    """Returns the summary provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.STD_STRING:
        return StdStringSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_OS_STRING:
        return StdOsStringSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_STR:
        return StdStrSummaryProvider(valobj, dict)

    if rust_type == RustType.STD_VEC:
        return SizeSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_VEC_DEQUE:
        return SizeSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_SLICE:
        return SizeSummaryProvider(valobj, dict)

    if rust_type == RustType.STD_HASH_MAP:
        return SizeSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_HASH_SET:
        return SizeSummaryProvider(valobj, dict)

    if rust_type == RustType.STD_RC:
        return StdRcSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_ARC:
        return StdRcSummaryProvider(valobj, dict)

    if rust_type == RustType.STD_REF:
        return StdRefSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_REF_MUT:
        return StdRefSummaryProvider(valobj, dict)
    if rust_type == RustType.STD_REF_CELL:
        return StdRefSummaryProvider(valobj, dict)

    if rust_type == RustType.STD_NONZERO_NUMBER:
        return StdNonZeroNumberSummaryProvider(valobj, dict)

    return ""


def synthetic_lookup(valobj, dict):
    # type: (SBValue, dict) -> object
    """Returns the synthetic provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.STRUCT:
        return StructSyntheticProvider(valobj, dict)
    if rust_type == RustType.STRUCT_VARIANT:
        return StructSyntheticProvider(valobj, dict, is_variant=True)
    if rust_type == RustType.TUPLE:
        return TupleSyntheticProvider(valobj, dict)
    if rust_type == RustType.TUPLE_VARIANT:
        return TupleSyntheticProvider(valobj, dict, is_variant=True)
    if rust_type == RustType.EMPTY:
        return EmptySyntheticProvider(valobj, dict)
    if rust_type == RustType.REGULAR_ENUM:
        discriminant = valobj.GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned()
        return synthetic_lookup(valobj.GetChildAtIndex(discriminant), dict)
    if rust_type == RustType.SINGLETON_ENUM:
        return synthetic_lookup(valobj.GetChildAtIndex(0), dict)

    if rust_type == RustType.STD_VEC:
        return StdVecSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_VEC_DEQUE:
        return StdVecDequeSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_SLICE:
        return StdSliceSyntheticProvider(valobj, dict)

    if rust_type == RustType.STD_HASH_MAP:
        if is_hashbrown_hashmap(valobj):
            return StdHashMapSyntheticProvider(valobj, dict)
        else:
            return StdOldHashMapSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_HASH_SET:
        hash_map = valobj.GetChildAtIndex(0)
        if is_hashbrown_hashmap(hash_map):
            return StdHashMapSyntheticProvider(valobj, dict, show_values=False)
        else:
            return StdOldHashMapSyntheticProvider(hash_map, dict, show_values=False)

    if rust_type == RustType.STD_RC:
        return StdRcSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_ARC:
        return StdRcSyntheticProvider(valobj, dict, is_atomic=True)

    if rust_type == RustType.STD_CELL:
        return StdCellSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_REF:
        return StdRefSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_REF_MUT:
        return StdRefSyntheticProvider(valobj, dict)
    if rust_type == RustType.STD_REF_CELL:
        return StdRefSyntheticProvider(valobj, dict, is_cell=True)

    return DefaultSynthteticProvider(valobj, dict)
