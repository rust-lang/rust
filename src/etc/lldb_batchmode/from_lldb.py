"""Contains LLDB conversion functions from LLDB's in-memory representations to the test classes
defined in `./common.py`.

We primarily interface with the following LLDB classes:

* [`SBValue`](https://lldb.llvm.org/python_api/lldb.SBValue.html)
* [`SBType`](https://lldb.llvm.org/python_api/lldb.SBType.html)
* [`SBTypeMember`](https://lldb.llvm.org/python_api/lldb.SBTypeMember.html)
"""

from struct import unpack, calcsize
from enum import Enum, IntFlag
from typing import Optional, Union

import lldb
import lldb_lookup

from .common import (
    BLESS,
    TARGET,
    Child,
    Field,
    Target,
    TargetData,
    Type,
    Variable,
)

HAS_FLOAT128: bool = getattr(lldb, "eBasicTypeFloat128", None) is not None

# We use the following lists to dynamically create the enums at run-time (they're used to print
# more meaningful error messages when basic_type and type_class don't match).
# It takes a few hundred microseconds at runtime to generate these lists, but it means we never have
# to upkeep version-specific flags. Since the underlying integers are what are stored and tested
# against, these don't affect (and are not affected by) the test data.
_lldb_type_classes = {
    k.removeprefix("eTypeClass"): v
    for k, v in lldb.__dict__.items()
    if k.startswith("eTypeClass")
}
_lldb_basic_types = {
    k.removeprefix("eBasicType"): v
    for k, v in lldb.__dict__.items()
    if k.startswith("eBasicType")
}


# We specify boundary=KEEP to tell python that values that aren't directly specified should still
# be formatted as if they're members of TypeClass (rather than throwing an exception)
class TypeClass(IntFlag):
    """Direct mapping of `lldb.eTypeClass` bitflags for convenience. Used to print a more meaningful
    error message when Type.type_class does not match.
    """

    # Enums create their members based on locals. We can access and modify the locals dict just like
    # any other. As gross as it is, this is canonical, per Python's own tutorial.
    # See: https://docs.python.org/3/howto/enum.html#timeperiod
    # The alternative is using the functional syntax, but that doesn't allow us to set boundary=KEEP
    vars().update(_lldb_type_classes)


class BasicType(Enum):
    """Direct mapping of `lldb.eBasicType` enumerations for convenience. Used to print a more
    meaningful error message when Type.basic_type does not match.
    """

    vars().update(_lldb_basic_types)


_UNSIGNED_INT_TYPES = {
    lldb.eBasicTypeUnsignedChar,
    lldb.eBasicTypeUnsignedShort,
    lldb.eBasicTypeUnsignedInt,
    lldb.eBasicTypeUnsignedLong,
    lldb.eBasicTypeUnsignedLongLong,
    lldb.eBasicTypeUnsignedInt128,
}

_FLOAT_TYPES = {
    lldb.eBasicTypeHalf,
    lldb.eBasicTypeFloat,
    lldb.eBasicTypeDouble,
}

if HAS_FLOAT128:
    _FLOAT_TYPES.add(lldb.eBasicTypeFloat128)

_SIZE_TO_FLOAT_FMT = {
    2: "e",
    4: "f",
    8: "d",
}

_SIZE_TO_INT_FMT = {
    1: "b",
    2: "h",
    4: "l",
    8: "q",
    # python doesn't have native support for u128 so we manually reconstruct it from 2 64-bit ints
    16: "qq",
}


def type_unpack_fmt(kind: int, size: int) -> str:
    # we can't just map directly from lldb.eBasicType -> format string because lldb.eBasicType types
    # aren't the same size on every target (even if targets have the same word size). e.g. On
    # windows, isize = lldb.eBasicTypeLongLong, on linux with identical hardware,
    # isize = lldb.eBasicTypeLong.
    # Conversely, python's struct.unpack format specifiers ARE consistenly sized.

    if kind in _FLOAT_TYPES:
        return _SIZE_TO_FLOAT_FMT[size]

    if kind == lldb.eBasicTypeBool:
        return "?"

    if kind == lldb.eBasicTypeChar32:
        return "4s"

    fmt = _SIZE_TO_INT_FMT[size]

    if kind in _UNSIGNED_INT_TYPES:
        fmt.upper()

    return fmt


def decode_primitive(valobj: lldb.SBValue) -> Union[int, float, bool, str]:
    data: lldb.SBData = valobj.GetData()

    type: lldb.SBType = valobj.GetType().GetCanonicalType()
    kind = type.GetBasicType()

    assert kind != lldb.eBasicTypeInvalid, f"{valobj.name} is not a primtive"

    is_big_endian = data.GetByteOrder() == lldb.eByteOrderBig

    buf = data.ReadRawData(lldb.SBError(), 0, data.GetByteSize())

    if is_big_endian or kind == lldb.eBasicTypeChar32:
        endian = ">"
    else:
        endian = "<"

    format = endian + type_unpack_fmt(kind, type.GetByteSize())
    # sanity check
    assert calcsize(format) == data.GetByteSize()

    got = unpack(format, buf)

    if kind == lldb.eBasicTypeChar32:
        got = got[0].decode("utf-32")
    elif kind in [lldb.eBasicTypeInt128, lldb.eBasicTypeUnsignedInt128]:
        # python doesn't have native support for u128 so we manually construct from 2 64-bit ints
        hi = got[0] if is_big_endian else got[1]
        lo = got[1] if is_big_endian else got[0]

        got = lo | (hi << 64)
    else:
        got = got[0]

    return got


def get_summary_or_value(valobj: lldb.SBValue) -> Optional[str]:
    """`SBValue.GetSummary` only prints summaries from summary providers. It returns `None` if there
    is no summary provider, rather than printing the default representation of the value. Often we
    want any printable representation at all, so this function falls back to `SBValue.GetValue`.
    That covers things like primitives and flat enums that typically don't have summary providers.
    """

    summary = valobj.GetSummary()
    if summary is None:
        return valobj.GetValue()

    return summary


def field_from_lldb(field: lldb.SBTypeMember) -> Field:
    if BLESS and not field.IsValid():
        raise Exception("Cannot bless invalid SBTypeMember object")

    return Field(field.GetName(), field.GetType().GetName(), field.GetOffsetInBytes())


def get_generics(ty: lldb.SBType, sbtarget: lldb.SBTarget) -> list[lldb.SBType]:
    """Platform-agnostic equivalent to `SBType.template_args`. `SBType`'s template functions do not
    work correctly with PDB debug info because PDB has no way to represent template parameters.

    Due to the DWARF spec using
    C++-centric terminology (e.g. `DW_TAG_template_type_parameter`), the following terms are
    interchangable:

    * template type param/arg <-> generic param
    * template value param/arg <-> const generic param

    The difference between "param" and "arg" is largely irrelevant for our purposes.
    Pre-parameterized types (e.g. `Vec<T>`, which could be parameterized to `Vec<u8>`, LLDB calls
    this "template specialization") are not reflected in the DWARF data at all, and are largely an
    LLDB/clang implementation detail that isn't directly exposed to us.
    """

    name = ty.GetName()
    # FIXME Rust doesn't output template *values* (the `10` in `ArrayVec<u8, 10>`), only
    # template args (the `u8` in `ArrayVec<u8, 10>`). That means these can possibly have
    # different results. That's not a big deal, I don't think anything in the std library uses
    # template values at the moment.
    # Eventually we can either change `get_template_args` to skip template values OR update
    # rustc to output them for DWARF debug info. Also, since it's target-specific behavior, it
    # shouldn't actually cause tests not to work.
    if TARGET == Target.WindowsMsvc:
        return [
            lldb_lookup.resolve_msvc_template_arg(x, sbtarget)
            for x in lldb_lookup.get_template_args(name)
        ]
    else:
        return [
            ty.GetTemplateArgumentType(i)
            for i in range(ty.GetNumberOfTemplateArguments())
        ]


def type_from_lldb(ty: lldb.SBType, sbtarget: lldb.SBTarget) -> Type:
    if BLESS and not ty.IsValid():
        raise Exception("Cannot bless invalid SBType object")

    generic_types = get_generics(ty, sbtarget)
    generics = [g.GetName() for g in generic_types]

    return Type(
        ty.GetByteSize(),
        ty.GetBasicType(),
        ty.GetTypeClass(),
        [field_from_lldb(ty.GetFieldAtIndex(i)) for i in range(ty.GetNumberOfFields())],
        generics,
    )


def child_from_lldb(child: lldb.SBValue) -> Child:
    if BLESS and not child.IsValid():
        raise Exception("Cannot bless invalid child")

    sbtype: lldb.SBType = child.GetType()

    if not sbtype.IsPointerType() and sbtype.GetBasicType() != lldb.eBasicTypeInvalid:
        value = decode_primitive(child)
    else:
        value = None

    children = [
        child_from_lldb(child.GetChildAtIndex(i)) for i in range(child.GetNumChildren())
    ]

    return Child(child.GetName(), child.GetType().GetName(), value, children)


def variable_from_lldb(var: lldb.SBValue) -> Variable:
    if BLESS and not var.IsValid():
        raise Exception("Cannot bless invalid SBValue object")

    sbtype = var.GetType()
    type_name = sbtype.GetName()

    pretty_type_name = var.GetDisplayTypeName()

    if pretty_type_name == type_name:
        pretty_type_name = None

    # We never want to store pointer values since they are expected to change from run to run.
    # For now, we also only want to record values for primitives. In the future, we may support
    # testing the `get_value` output from syntheticproviders, but the current visualizers do not
    # implement this so it isn't urgent.
    if not sbtype.IsPointerType() and sbtype.GetBasicType() != lldb.eBasicTypeInvalid:
        value = decode_primitive(var)
    else:
        value = None

    if (synth := var.GetTypeSynthetic()).IsValid():
        synthetic = synth.GetData()
        if synthetic is not None:
            synthetic = synthetic.strip()
    else:
        synthetic = None

    if (summ := var.GetTypeSummary()).IsValid():
        summary = summ.GetData()
        if summary is not None:
            summary = summary.strip()
    else:
        summary = None

    if (fmt := var.GetTypeFormat()).IsValid():
        format = fmt.GetFormat()
    else:
        format = None

    pretty_print = get_summary_or_value(var)

    children = [
        child_from_lldb(var.GetChildAtIndex(i)) for i in range(var.GetNumChildren())
    ]

    return Variable(
        type_name,
        pretty_type_name,
        pretty_print,
        value,
        synthetic,
        summary,
        format,
        children,
    )


def bless_variable(
    target_data: TargetData, var_name: str, breakpoint_idx: int, frame: lldb.SBFrame
):
    """Updates the given `TargetData` with data generated from the given variable at the given
    breakpoint. This function **does not** write to the input file. Please see
    `TargetData.save_blessing` for more info on when and how to save the data.
    """

    valobj = frame.FindVariable(var_name)
    if not valobj.IsValid():
        # FIXME (todo) error handling
        raise Exception(f"<bless error: Cannot find variable {var_name}>")

    # HACK it's obviously not ideal to output empty breakpoints, but it will be somewhat rare for it
    # to happen (you would need a breakpoint with repr -> breakpoint without repr -> breakpoint
    # with repr). In the more common case (e.g. 1 breakpoint, sequential breakpoints all with repr
    # commands), this saves a lot more space than converting all TargetData.breakpoints to
    # `dict[int,...]`
    while len(target_data.breakpoints) <= breakpoint_idx:
        target_data.breakpoints.append({})

    var_data = variable_from_lldb(valobj)
    target_data.breakpoints[breakpoint_idx][var_name] = var_data

    # We also need to bless the types of the valobj's children, as they may not appear in the type
    # or fields.
    target = valobj.GetTarget()

    work_list = [valobj]
    while len(work_list) != 0:
        obj = work_list.pop()
        work_list.extend([obj.GetChildAtIndex(i) for i in range(obj.GetNumChildren())])

        bless_type(target_data, obj.GetType(), target)


def bless_type(target_data: TargetData, type: lldb.SBType, sbtarget: lldb.SBTarget):
    """Recursively adds the type and all types of its fields to the `target_data.types` mapping"""

    t_name = type.GetName()
    t_data = type_from_lldb(type, sbtarget)
    if t_name in target_data.types:
        # If the type already exists in the type map, we don't need to process any further. We do
        # need to check that the type data is actually identical to its mapping before moving on.
        # It shouldn't ever be different, but better safe than sorry.
        import pprint

        assert (
            target_data.types[t_name] == t_data
        ), f"old: {pprint.pformat(target_data.types[t_name])}\nnew: {pprint.pformat(t_data)}"
        return

    print(f"blessing type: {t_name}")

    # We need to add this type first just in case the type contains itself.
    target_data.types[t_name] = t_data

    # For types we haven't seen, we need to recursively handle the types of all of the fields and
    # generics
    for i in range(type.GetNumberOfFields()):
        field = type.GetFieldAtIndex(i)
        f_type = field.GetType()
        f_type_name = f_type.GetName()

        if f_type_name not in target_data.types:
            bless_type(target_data, f_type, sbtarget)

    for generic in get_generics(type, sbtarget):
        # FIXME the purpose of this check is to gracefully handle generic *values* (e.g. the `2` in
        # `ArrayVec<u8, 2>`) that slipped through the msvc template arg handling. At some point,
        # `lldb_providers.get_template_args` should be made to output whether or not a template arg
        # is a value, but for now this should be fine.
        if generic.IsValid():
            bless_type(target_data, generic, sbtarget)
