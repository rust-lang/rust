import lldb

import lldb_providers as providers
from rust_types import RustType, classify_struct, classify_union


def classify_rust_type(type: lldb.SBType) -> str:
    type_class = type.GetTypeClass()
    if type_class == lldb.eTypeClassStruct:
        return classify_struct(type.name, type.fields)
    if type_class == lldb.eTypeClassUnion:
        return classify_union(type.fields)

    return RustType.OTHER


def summary_lookup(valobj: lldb.SBValue, _dict: providers.LLDBOpaque) -> str:
    """Returns the summary provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.STD_STRING:
        return providers.StdStringSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_OS_STRING:
        return providers.StdOsStringSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_STR:
        return providers.StdStrSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_VEC:
        return providers.SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_VEC_DEQUE:
        return providers.SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_SLICE:
        return providers.SizeSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_HASH_MAP:
        return providers.SizeSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_HASH_SET:
        return providers.SizeSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_RC:
        return providers.StdRcSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_ARC:
        return providers.StdRcSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_REF:
        return providers.StdRefSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_MUT:
        return providers.StdRefSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_CELL:
        return providers.StdRefSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_NONZERO_NUMBER:
        return providers.StdNonZeroNumberSummaryProvider(valobj, _dict)

    if rust_type == RustType.STD_PATHBUF:
        return providers.StdPathBufSummaryProvider(valobj, _dict)
    if rust_type == RustType.STD_PATH:
        return providers.StdPathSummaryProvider(valobj, _dict)

    return ""


def synthetic_lookup(valobj: lldb.SBValue, _dict: providers.LLDBOpaque) -> object:
    """Returns the synthetic provider for the given value"""
    rust_type = classify_rust_type(valobj.GetType())

    if rust_type == RustType.STRUCT:
        return providers.StructSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STRUCT_VARIANT:
        return providers.StructSyntheticProvider(valobj, _dict, is_variant=True)
    if rust_type == RustType.TUPLE:
        return providers.TupleSyntheticProvider(valobj, _dict)
    if rust_type == RustType.TUPLE_VARIANT:
        return providers.TupleSyntheticProvider(valobj, _dict, is_variant=True)
    if rust_type == RustType.EMPTY:
        return providers.EmptySyntheticProvider(valobj, _dict)
    if rust_type == RustType.REGULAR_ENUM:
        discriminant = valobj.GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned()
        return synthetic_lookup(valobj.GetChildAtIndex(discriminant), _dict)
    if rust_type == RustType.SINGLETON_ENUM:
        return synthetic_lookup(valobj.GetChildAtIndex(0), _dict)
    if rust_type == RustType.ENUM:
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

        return providers.ClangEncodedEnumProvider(valobj, _dict)
    if rust_type == RustType.STD_VEC:
        return providers.StdVecSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_VEC_DEQUE:
        return providers.StdVecDequeSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_SLICE or rust_type == RustType.STD_STR:
        return providers.StdSliceSyntheticProvider(valobj, _dict)

    if rust_type == RustType.STD_HASH_MAP:
        return providers.StdHashMapSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_HASH_SET:
        return providers.StdHashMapSyntheticProvider(valobj, _dict, show_values=False)

    if rust_type == RustType.STD_RC:
        return providers.StdRcSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_ARC:
        return providers.StdRcSyntheticProvider(valobj, _dict, is_atomic=True)

    if rust_type == RustType.STD_CELL:
        return providers.StdCellSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_REF:
        return providers.StdRefSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_MUT:
        return providers.StdRefSyntheticProvider(valobj, _dict)
    if rust_type == RustType.STD_REF_CELL:
        return providers.StdRefSyntheticProvider(valobj, _dict, is_cell=True)

    return providers.DefaultSyntheticProvider(valobj, _dict)
