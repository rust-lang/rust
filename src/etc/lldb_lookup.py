from __future__ import annotations
from typing import Dict

import lldb

from lldb_providers import *
from rust_types import RustType, classify_struct, classify_union


# BACKCOMPAT: rust 1.35
def is_hashbrown_hashmap(hash_map: lldb.SBValue) -> bool:
    return len(hash_map.type.fields) == 1


def classify_rust_type(type: lldb.SBType) -> RustType:
    if type.IsPointerType():
        type = type.GetPointeeType()

    type_class = type.GetTypeClass()
    if type_class == lldb.eTypeClassStruct:
        return classify_struct(type.name, type.fields)
    if type_class == lldb.eTypeClassUnion:
        return classify_union(type.fields)

    return RustType.Other


def summary_lookup(valobj: lldb.SBValue, _dict: LLDBOpaque) -> str:
    """Returns the summary provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.StdString:
        return StdStringSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdOsString:
        return StdOsStringSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdStr:
        return StdStrSummaryProvider(valobj, _dict)

    if rust_type == RustType.StdVec:
        return SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdVecDeque:
        return SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdSlice:
        return SizeSummaryProvider(valobj, _dict)

    if rust_type == RustType.StdHashMap:
        return SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdHashSet:
        return SizeSummaryProvider(valobj, _dict)

    if rust_type == RustType.StdRc:
        return StdRcSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdArc:
        return StdRcSummaryProvider(valobj, _dict)

    if rust_type == RustType.StdRef:
        return StdRefSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdRefMut:
        return StdRefSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdRefCell:
        return StdRefSummaryProvider(valobj, _dict)

    if rust_type == RustType.StdNonZeroNumber:
        return StdNonZeroNumberSummaryProvider(valobj, _dict)

    if rust_type == RustType.StdPathBuf:
        return StdPathBufSummaryProvider(valobj, _dict)
    if rust_type == RustType.StdPath:
        return StdPathSummaryProvider(valobj, _dict)

    return ""


def synthetic_lookup(valobj: lldb.SBValue, _dict: LLDBOpaque) -> object:
    """Returns the synthetic provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.Struct:
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
        return ClangEncodedEnumProvider(valobj, _dict)
    if rust_type == RustType.StdVec:
        return StdVecSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdVecDeque:
        return StdVecDequeSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdSlice or rust_type == RustType.StdStr:
        return StdSliceSyntheticProvider(valobj, _dict)

    if rust_type == RustType.StdHashMap:
        if is_hashbrown_hashmap(valobj):
            return StdHashMapSyntheticProvider(valobj, _dict)
        else:
            return StdOldHashMapSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdHashSet:
        hash_map = valobj.GetChildAtIndex(0)
        if is_hashbrown_hashmap(hash_map):
            return StdHashMapSyntheticProvider(valobj, _dict, show_values=False)
        else:
            return StdOldHashMapSyntheticProvider(hash_map, _dict, show_values=False)

    if rust_type == RustType.StdRc:
        return StdRcSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdArc:
        return StdRcSyntheticProvider(valobj, _dict, is_atomic=True)

    if rust_type == RustType.StdCell:
        return StdCellSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdRef:
        return StdRefSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdRefMut:
        return StdRefSyntheticProvider(valobj, _dict)
    if rust_type == RustType.StdRefCell:
        return StdRefSyntheticProvider(valobj, _dict, is_cell=True)

    return DefaultSyntheticProvider(valobj, _dict)
