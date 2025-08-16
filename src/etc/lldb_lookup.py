import lldb

from lldb_providers import *
from rust_types import RustType, classify_struct, classify_union


# BACKCOMPAT: rust 1.35
def is_hashbrown_hashmap(hash_map: lldb.SBValue) -> bool:
    return len(hash_map.type.fields) == 1


def classify_rust_type(type: lldb.SBType) -> str:
    if type.IsPointerType():
        type = type.GetPointeeType()

    type_class = type.GetTypeClass()
    if type_class == lldb.eTypeClassStruct:
        return classify_struct(type.name, type.fields)
    if type_class == lldb.eTypeClassUnion:
        return classify_union(type.fields)

    return RustType.OTHER


def summary_lookup(valobj: lldb.SBValue, _dict: LLDBOpaque) -> str:
    """Returns the summary provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.STD_STRING:
        return StdStringSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_OS_STRING:
        return StdOsStringSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_STR:
        return StdStrSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_VEC:
        return SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_VEC_DEQUE:
        return SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_SLICE:
        return SizeSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_HASH_MAP:
        return SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_HASH_SET:
        return SizeSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_RC:
        return StdRcSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_ARC:
        return StdRcSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_REF:
        return StdRefSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_MUT:
        return StdRefSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_CELL:
        return StdRefSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_NONZERO_NUMBER:
        return StdNonZeroNumberSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_PATHBUF:
        return StdPathBufSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_PATH:
        return StdPathSummaryProvider(valobj, _dict)

    return ""


def synthetic_lookup(valobj: lldb.SBValue, _dict: LLDBOpaque) -> object:
    """Returns the synthetic provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.STRUCT:
        return StructSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STRUCT_VARIANT:
        return StructSyntheticProvider(valobj, _dict, is_variant=True)
    if rust_type == RustType.TUPLE:
        return TupleSyntheticProvider(valobj, _dict)
    if rust_type == RustType.TUPLE_VARIANT:
        return TupleSyntheticProvider(valobj, _dict, is_variant=True)
    if rust_type == RustType.EMPTY:
        return EmptySyntheticProvider(valobj, _dict)
    if rust_type == RustType.REGULAR_ENUM:
        discriminant = valobj.GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned()
        return synthetic_lookup(valobj.GetChildAtIndex(discriminant), _dict)
    if rust_type == RustType.SINGLETON_ENUM:
        return synthetic_lookup(valobj.GetChildAtIndex(0), _dict)
    if rust_type == RustType.ENUM:
        return ClangEncodedEnumProvider(valobj, _dict)
    if rust_type == RustType.STD_VEC:
        return StdVecSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_VEC_DEQUE:
        return StdVecDequeSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_SLICE or rust_type == RustType.STD_STR:
        return StdSliceSyntheticProvider(valobj, _dict)

    if rust_type == RustType.STD_HASH_MAP:
        if is_hashbrown_hashmap(valobj):
            return StdHashMapSyntheticProvider(valobj, _dict)
        else:
            return StdOldHashMapSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_HASH_SET:
        hash_map = valobj.GetChildAtIndex(0)
        if is_hashbrown_hashmap(hash_map):
            return StdHashMapSyntheticProvider(valobj, _dict, show_values=False)
        else:
            return StdOldHashMapSyntheticProvider(hash_map, _dict, show_values=False)

    if rust_type == RustType.STD_RC:
        return StdRcSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_ARC:
        return StdRcSyntheticProvider(valobj, _dict, is_atomic=True)

    if rust_type == RustType.STD_CELL:
        return StdCellSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_REF:
        return StdRefSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_MUT:
        return StdRefSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_CELL:
        return StdRefSyntheticProvider(valobj, _dict, is_cell=True)

    return DefaultSyntheticProvider(valobj, _dict)
