from __future__ import annotations
import re
import sys
from typing import List, TYPE_CHECKING

from lldb import (
    SBData,
    SBError,
    eBasicTypeLong,
    eBasicTypeUnsignedLong,
    eBasicTypeUnsignedChar,
    eFormatChar,
)

if TYPE_CHECKING:
    from lldb import SBValue, SBType, SBTypeStaticField

# from lldb.formatters import Logger

####################################################################################################
# This file contains two kinds of pretty-printers: summary and synthetic.
#
# Important classes from LLDB module:
#   SBValue: the value of a variable, a register, or an expression
#   SBType:  the data type; each SBValue has a corresponding SBType
#
# Summary provider is a function with the type `(SBValue, dict) -> str`.
#   The first parameter is the object encapsulating the actual variable being displayed;
#   The second parameter is an internal support parameter used by LLDB, and you should not touch it.
#
# Synthetic children is the way to provide a children-based representation of the object's value.
# Synthetic provider is a class that implements the following interface:
#
#     class SyntheticChildrenProvider:
#         def __init__(self, SBValue, dict)
#         def num_children(self)
#         def get_child_index(self, str)
#         def get_child_at_index(self, int)
#         def update(self)
#         def has_children(self)
#         def get_value(self)
#
#
# You can find more information and examples here:
#   1. https://lldb.llvm.org/varformats.html
#   2. https://lldb.llvm.org/use/python-reference.html
#   3. https://github.com/llvm/llvm-project/blob/llvmorg-8.0.1/lldb/www/python_reference/lldb.formatters.cpp-pysrc.html
#   4. https://github.com/llvm-mirror/lldb/tree/master/examples/summaries/cocoa
####################################################################################################

PY3 = sys.version_info[0] == 3


class LLDBOpaque:
    """
    An marker type for use in type hints to denote LLDB bookkeeping variables. Values marked with
    this type should never be used except when passing as an argument to an LLDB function.
    """


class ValueBuilder:
    def __init__(self, valobj: SBValue):
        self.valobj = valobj
        process = valobj.GetProcess()
        self.endianness = process.GetByteOrder()
        self.pointer_size = process.GetAddressByteSize()

    def from_int(self, name: str, value: int) -> SBValue:
        type = self.valobj.GetType().GetBasicType(eBasicTypeLong)
        data = SBData.CreateDataFromSInt64Array(
            self.endianness, self.pointer_size, [value]
        )
        return self.valobj.CreateValueFromData(name, data, type)

    def from_uint(self, name: str, value: int) -> SBValue:
        type = self.valobj.GetType().GetBasicType(eBasicTypeUnsignedLong)
        data = SBData.CreateDataFromUInt64Array(
            self.endianness, self.pointer_size, [value]
        )
        return self.valobj.CreateValueFromData(name, data, type)


def unwrap_unique_or_non_null(unique_or_nonnull: SBValue) -> SBValue:
    # BACKCOMPAT: rust 1.32
    # https://github.com/rust-lang/rust/commit/7a0911528058e87d22ea305695f4047572c5e067
    # BACKCOMPAT: rust 1.60
    # https://github.com/rust-lang/rust/commit/2a91eeac1a2d27dd3de1bf55515d765da20fd86f
    ptr = unique_or_nonnull.GetChildMemberWithName("pointer")
    return ptr if ptr.TypeIsPointerType() else ptr.GetChildAtIndex(0)


class DefaultSyntheticProvider:
    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        # logger = Logger.Logger()
        # logger >> "Default synthetic provider for " + str(valobj.GetName())
        self.valobj = valobj

    def num_children(self) -> int:
        return self.valobj.GetNumChildren()

    def get_child_index(self, name: str) -> int:
        return self.valobj.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index: int) -> SBValue:
        return self.valobj.GetChildAtIndex(index)

    def update(self):
        pass

    def has_children(self) -> bool:
        return self.valobj.MightHaveChildren()


class EmptySyntheticProvider:
    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        # logger = Logger.Logger()
        # logger >> "[EmptySyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj

    def num_children(self) -> int:
        return 0

    def get_child_index(self, name: str) -> int:
        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        return None

    def update(self):
        pass

    def has_children(self) -> bool:
        return False


def get_template_args(type_name: str) -> list[str]:
    """
    Takes a type name `T<A, tuple$<B, C>, D>` and returns a list of its generic args
    `["A", "tuple$<B, C>", "D"]`.

    String-based replacement for LLDB's `SBType.template_args`, as LLDB is currently unable to
    populate this field for targets with PDB debug info. Also useful for manually altering the type
    name of generics (e.g. `Vec<ref$<str$>` -> `Vec<&str>`).

    Each element of the returned list can be looked up for its `SBType` value via
    `SBTarget.FindFirstType()`
    """
    params = []
    level = 0
    start = 0
    for i, c in enumerate(type_name):
        if c == "<":
            level += 1
            if level == 1:
                start = i + 1
        elif c == ">":
            level -= 1
            if level == 0:
                params.append(type_name[start:i].strip())
        elif c == "," and level == 1:
            params.append(type_name[start:i].strip())
            start = i + 1
    return params


def SizeSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    return "size=" + str(valobj.GetNumChildren())


def vec_to_string(vec: SBValue) -> str:
    length = vec.GetNumChildren()
    chars = [vec.GetChildAtIndex(i).GetValueAsUnsigned() for i in range(length)]
    return (
        bytes(chars).decode(errors="replace")
        if PY3
        else "".join(chr(char) for char in chars)
    )


def StdStringSummaryProvider(valobj, dict):
    inner_vec = (
        valobj.GetNonSyntheticValue()
        .GetChildMemberWithName("vec")
        .GetNonSyntheticValue()
    )

    pointer = (
        inner_vec.GetChildMemberWithName("buf")
        .GetChildMemberWithName("inner")
        .GetChildMemberWithName("ptr")
        .GetChildMemberWithName("pointer")
        .GetChildMemberWithName("pointer")
    )

    length = inner_vec.GetChildMemberWithName("len").GetValueAsUnsigned()

    if length <= 0:
        return '""'
    error = SBError()
    process = pointer.GetProcess()
    data = process.ReadMemory(pointer.GetValueAsUnsigned(), length, error)
    if error.Success():
        return '"' + data.decode("utf8", "replace") + '"'
    else:
        raise Exception("ReadMemory error: %s", error.GetCString())


def StdOsStringSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    # logger = Logger.Logger()
    # logger >> "[StdOsStringSummaryProvider] for " + str(valobj.GetName())
    buf = valobj.GetChildAtIndex(0).GetChildAtIndex(0)
    is_windows = "Wtf8Buf" in buf.type.name
    vec = buf.GetChildAtIndex(0) if is_windows else buf
    return '"%s"' % vec_to_string(vec)


def StdStrSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    # logger = Logger.Logger()
    # logger >> "[StdStrSummaryProvider] for " + str(valobj.GetName())

    # the code below assumes non-synthetic value, this makes sure the assumption holds
    valobj = valobj.GetNonSyntheticValue()

    length = valobj.GetChildMemberWithName("length").GetValueAsUnsigned()
    if length == 0:
        return '""'

    data_ptr = valobj.GetChildMemberWithName("data_ptr")

    start = data_ptr.GetValueAsUnsigned()
    error = SBError()
    process = data_ptr.GetProcess()
    data = process.ReadMemory(start, length, error)
    data = data.decode(encoding="UTF-8") if PY3 else data
    return '"%s"' % data


def StdPathBufSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    # logger = Logger.Logger()
    # logger >> "[StdPathBufSummaryProvider] for " + str(valobj.GetName())
    return StdOsStringSummaryProvider(valobj.GetChildMemberWithName("inner"), _dict)


def StdPathSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    # logger = Logger.Logger()
    # logger >> "[StdPathSummaryProvider] for " + str(valobj.GetName())
    length = valobj.GetChildMemberWithName("length").GetValueAsUnsigned()
    if length == 0:
        return '""'

    data_ptr = valobj.GetChildMemberWithName("data_ptr")

    start = data_ptr.GetValueAsUnsigned()
    error = SBError()
    process = data_ptr.GetProcess()
    data = process.ReadMemory(start, length, error)
    if PY3:
        try:
            data = data.decode(encoding="UTF-8")
        except UnicodeDecodeError:
            return "%r" % data
    return '"%s"' % data


def sequence_formatter(output: str, valobj: SBValue, _dict: LLDBOpaque):
    length: int = valobj.GetNumChildren()

    long: bool = False
    for i in range(0, length):
        if len(output) > 32:
            long = True
            break

        child: SBValue = valobj.GetChildAtIndex(i)

        summary = child.summary
        if summary is None:
            summary = child.value
            if summary is None:
                summary = "{...}"
        output += f"{summary}, "
    if long:
        output = f"(len: {length}) " + output + "..."
    else:
        output = output[:-2]

    return output


class StructSyntheticProvider:
    """Pretty-printer for structs and struct enum variants"""

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque, is_variant: bool = False):
        # logger = Logger.Logger()
        self.valobj = valobj
        self.is_variant = is_variant
        self.type = valobj.GetType()
        self.fields = {}

        if is_variant:
            self.fields_count = self.type.GetNumberOfFields() - 1
            real_fields = self.type.fields[1:]
        else:
            self.fields_count = self.type.GetNumberOfFields()
            real_fields = self.type.fields

        for number, field in enumerate(real_fields):
            self.fields[field.name] = number

    def num_children(self) -> int:
        return self.fields_count

    def get_child_index(self, name: str) -> int:
        return self.fields.get(name, -1)

    def get_child_at_index(self, index: int) -> SBValue:
        if self.is_variant:
            field = self.type.GetFieldAtIndex(index + 1)
        else:
            field = self.type.GetFieldAtIndex(index)
        return self.valobj.GetChildMemberWithName(field.name)

    def update(self):
        # type: () -> None
        pass

    def has_children(self) -> bool:
        return True


class StdStringSyntheticProvider:
    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj
        self.update()

    def update(self):
        inner_vec = self.valobj.GetChildMemberWithName("vec").GetNonSyntheticValue()
        self.data_ptr = (
            inner_vec.GetChildMemberWithName("buf")
            .GetChildMemberWithName("inner")
            .GetChildMemberWithName("ptr")
            .GetChildMemberWithName("pointer")
            .GetChildMemberWithName("pointer")
        )
        self.length = inner_vec.GetChildMemberWithName("len").GetValueAsUnsigned()
        self.element_type = self.data_ptr.GetType().GetPointeeType()

    def has_children(self) -> bool:
        return True

    def num_children(self) -> int:
        return self.length

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)

        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if not 0 <= index < self.length:
            return None
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + index
        element = self.data_ptr.CreateValueFromAddress(
            f"[{index}]", address, self.element_type
        )
        element.SetFormat(eFormatChar)
        return element


class MSVCStrSyntheticProvider:
    __slots__ = ["valobj", "data_ptr", "length"]

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj
        self.update()

    def update(self):
        self.data_ptr = self.valobj.GetChildMemberWithName("data_ptr")
        self.length = self.valobj.GetChildMemberWithName("length").GetValueAsUnsigned()

    def has_children(self) -> bool:
        return True

    def num_children(self) -> int:
        return self.length

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)

        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if not 0 <= index < self.length:
            return None
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + index
        element = self.data_ptr.CreateValueFromAddress(
            f"[{index}]", address, self.data_ptr.GetType().GetPointeeType()
        )
        return element

    def get_type_name(self):
        if self.valobj.GetTypeName().startswith("ref_mut"):
            return "&mut str"
        else:
            return "&str"


def _getVariantName(variant) -> str:
    """
    Since the enum variant's type name is in the form `TheEnumName::TheVariantName$Variant`,
    we can extract `TheVariantName` from it for display purpose.
    """
    s = variant.GetType().GetName()
    match = re.search(r"::([^:]+)\$Variant$", s)
    return match.group(1) if match else ""


class ClangEncodedEnumProvider:
    """Pretty-printer for 'clang-encoded' enums support implemented in LLDB"""

    DISCRIMINANT_MEMBER_NAME = "$discr$"
    VALUE_MEMBER_NAME = "value"

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj
        self.update()

    def has_children(self) -> bool:
        return True

    def num_children(self) -> int:
        return 1

    def get_child_index(self, _name: str) -> int:
        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if index == 0:
            value = self.variant.GetChildMemberWithName(
                ClangEncodedEnumProvider.VALUE_MEMBER_NAME
            )
            return value.CreateChildAtOffset(
                _getVariantName(self.variant), 0, value.GetType()
            )
        return None

    def update(self):
        all_variants = self.valobj.GetChildAtIndex(0)
        index = self._getCurrentVariantIndex(all_variants)
        self.variant = all_variants.GetChildAtIndex(index)

    def _getCurrentVariantIndex(self, all_variants: SBValue) -> int:
        default_index = 0
        for i in range(all_variants.GetNumChildren()):
            variant = all_variants.GetChildAtIndex(i)
            discr = variant.GetChildMemberWithName(
                ClangEncodedEnumProvider.DISCRIMINANT_MEMBER_NAME
            )
            if discr.IsValid():
                discr_unsigned_value = discr.GetValueAsUnsigned()
                if variant.GetName() == f"$variant${discr_unsigned_value}":
                    return i
            else:
                default_index = i
        return default_index


class MSVCEnumSyntheticProvider:
    """
    Synthetic provider for sum-type enums on MSVC. For a detailed explanation of the internals,
    see:

    https://github.com/rust-lang/rust/blob/master/compiler/rustc_codegen_llvm/src/debuginfo/metadata/enums/cpp_like.rs
    """

    __slots__ = ["valobj", "variant", "value"]

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj
        self.variant: SBValue
        self.value: SBValue
        self.update()

    def update(self):
        tag: SBValue = self.valobj.GetChildMemberWithName("tag")

        if tag.IsValid():
            tag: int = tag.GetValueAsUnsigned()
            for child in self.valobj.GetNonSyntheticValue().children:
                if not child.name.startswith("variant"):
                    continue

                variant_type: SBType = child.GetType()
                try:
                    exact: SBTypeStaticField = variant_type.GetStaticFieldWithName(
                        "DISCR_EXACT"
                    )
                except AttributeError:
                    # LLDB versions prior to 19.0.0 do not have the `SBTypeGetStaticField` API.
                    # With current DI generation there's not a great way to provide a "best effort"
                    # evaluation either, so we just return the object itself with no further
                    # attempts to inspect the type information
                    self.variant = self.valobj
                    self.value = self.valobj
                    return

                if exact.IsValid():
                    discr: int = exact.GetConstantValue(
                        self.valobj.target
                    ).GetValueAsUnsigned()
                    if tag == discr:
                        self.variant = child
                        self.value = child.GetChildMemberWithName(
                            "value"
                        ).GetSyntheticValue()
                        return
                else:  # if invalid, DISCR must be a range
                    begin: int = (
                        variant_type.GetStaticFieldWithName("DISCR_BEGIN")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )
                    end: int = (
                        variant_type.GetStaticFieldWithName("DISCR_END")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )

                    # begin isn't necessarily smaller than end, so we must test for both cases
                    if begin < end:
                        if begin <= tag <= end:
                            self.variant = child
                            self.value = child.GetChildMemberWithName(
                                "value"
                            ).GetSyntheticValue()
                            return
                    else:
                        if tag >= begin or tag <= end:
                            self.variant = child
                            self.value = child.GetChildMemberWithName(
                                "value"
                            ).GetSyntheticValue()
                            return
        else:  # if invalid, tag is a 128 bit value
            tag_lo: int = self.valobj.GetChildMemberWithName(
                "tag128_lo"
            ).GetValueAsUnsigned()
            tag_hi: int = self.valobj.GetChildMemberWithName(
                "tag128_hi"
            ).GetValueAsUnsigned()

            tag: int = (tag_hi << 64) | tag_lo

            for child in self.valobj.GetNonSyntheticValue().children:
                if not child.name.startswith("variant"):
                    continue

                variant_type: SBType = child.GetType()
                exact_lo: SBTypeStaticField = variant_type.GetStaticFieldWithName(
                    "DISCR128_EXACT_LO"
                )

                if exact_lo.IsValid():
                    exact_lo: int = exact_lo.GetConstantValue(
                        self.valobj.target
                    ).GetValueAsUnsigned()
                    exact_hi: int = (
                        variant_type.GetStaticFieldWithName("DISCR128_EXACT_HI")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )

                    discr: int = (exact_hi << 64) | exact_lo
                    if tag == discr:
                        self.variant = child
                        self.value = child.GetChildMemberWithName(
                            "value"
                        ).GetSyntheticValue()
                        return
                else:  # if invalid, DISCR must be a range
                    begin_lo: int = (
                        variant_type.GetStaticFieldWithName("DISCR128_BEGIN_LO")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )
                    begin_hi: int = (
                        variant_type.GetStaticFieldWithName("DISCR128_BEGIN_HI")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )

                    end_lo: int = (
                        variant_type.GetStaticFieldWithName("DISCR128_END_LO")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )
                    end_hi: int = (
                        variant_type.GetStaticFieldWithName("DISCR128_END_HI")
                        .GetConstantValue(self.valobj.target)
                        .GetValueAsUnsigned()
                    )

                    begin = (begin_hi << 64) | begin_lo
                    end = (end_hi << 64) | end_lo

                    # begin isn't necessarily smaller than end, so we must test for both cases
                    if begin < end:
                        if begin <= tag <= end:
                            self.variant = child
                            self.value = child.GetChildMemberWithName(
                                "value"
                            ).GetSyntheticValue()
                            return
                    else:
                        if tag >= begin or tag <= end:
                            self.variant = child
                            self.value = child.GetChildMemberWithName(
                                "value"
                            ).GetSyntheticValue()
                            return

    def num_children(self) -> int:
        return self.value.GetNumChildren()

    def get_child_index(self, name: str) -> int:
        return self.value.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index: int) -> SBValue:
        return self.value.GetChildAtIndex(index)

    def has_children(self) -> bool:
        return self.value.MightHaveChildren()

    def get_type_name(self) -> str:
        name = self.valobj.GetTypeName()
        # remove "enum2$<", str.removeprefix() is python 3.9+
        name = name[7:]

        # MSVC misinterprets ">>" as a shift operator, so spaces are inserted by rust to
        # avoid that
        if name.endswith(" >"):
            name = name[:-2]
        elif name.endswith(">"):
            name = name[:-1]

        return name


def MSVCEnumSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    enum_synth = MSVCEnumSyntheticProvider(valobj.GetNonSyntheticValue(), _dict)
    variant_names: SBType = valobj.target.FindFirstType(
        f"{enum_synth.valobj.GetTypeName()}::VariantNames"
    )
    try:
        name_idx = (
            enum_synth.variant.GetType()
            .GetStaticFieldWithName("NAME")
            .GetConstantValue(valobj.target)
            .GetValueAsUnsigned()
        )
    except AttributeError:
        # LLDB versions prior to 19 do not have the `SBTypeGetStaticField` API, and have no way
        # to determine the value based on the tag field.
        tag: SBValue = valobj.GetChildMemberWithName("tag")

        if tag.IsValid():
            discr: int = tag.GetValueAsUnsigned()
            return "".join(["{tag = ", str(tag.unsigned), "}"])
        else:
            tag_lo: int = valobj.GetChildMemberWithName(
                "tag128_lo"
            ).GetValueAsUnsigned()
            tag_hi: int = valobj.GetChildMemberWithName(
                "tag128_hi"
            ).GetValueAsUnsigned()

            discr: int = (tag_hi << 64) | tag_lo

        return "".join(["{tag = ", str(discr), "}"])

    name: str = variant_names.enum_members[name_idx].name

    if enum_synth.num_children() == 0:
        return name

    child_name: str = enum_synth.value.GetChildAtIndex(0).name
    if child_name == "0" or child_name == "__0":
        # enum variant is a tuple struct
        return name + TupleSummaryProvider(enum_synth.value, _dict)
    else:
        # enum variant is a regular struct
        var_list = (
            str(enum_synth.value.GetNonSyntheticValue()).split("= ", 1)[1].splitlines()
        )
        vars = [x.strip() for x in var_list if x not in ("{", "}")]
        if vars[0][0] == "(":
            vars[0] = vars[0][1:]
        if vars[-1][-1] == ")":
            vars[-1] = vars[-1][:-1]

        return f"{name}{{{', '.join(vars)}}}"


class TupleSyntheticProvider:
    """Pretty-printer for tuples and tuple enum variants"""

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque, is_variant: bool = False):
        # logger = Logger.Logger()
        self.valobj = valobj
        self.is_variant = is_variant
        self.type = valobj.GetType()

        if is_variant:
            self.size = self.type.GetNumberOfFields() - 1
        else:
            self.size = self.type.GetNumberOfFields()

    def num_children(self) -> int:
        return self.size

    def get_child_index(self, name: str) -> int:
        if name.isdigit():
            return int(name)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if self.is_variant:
            field = self.type.GetFieldAtIndex(index + 1)
        else:
            field = self.type.GetFieldAtIndex(index)
        element = self.valobj.GetChildMemberWithName(field.name)
        return self.valobj.CreateValueFromData(
            str(index), element.GetData(), element.GetType()
        )

    def update(self):
        pass

    def has_children(self) -> bool:
        return True


class MSVCTupleSyntheticProvider:
    __slots__ = ["valobj"]

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj

    def num_children(self) -> int:
        return self.valobj.GetNumChildren()

    def get_child_index(self, name: str) -> int:
        return self.valobj.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index: int) -> SBValue:
        child: SBValue = self.valobj.GetChildAtIndex(index)
        return child.CreateChildAtOffset(str(index), 0, child.GetType())

    def update(self):
        pass

    def has_children(self) -> bool:
        return self.valobj.MightHaveChildren()

    def get_type_name(self) -> str:
        name = self.valobj.GetTypeName()
        # remove "tuple$<" and ">", str.removeprefix and str.removesuffix require python 3.9+
        name = name[7:-1]
        return "(" + name + ")"


def TupleSummaryProvider(valobj: SBValue, _dict: LLDBOpaque):
    output: List[str] = []

    for i in range(0, valobj.GetNumChildren()):
        child: SBValue = valobj.GetChildAtIndex(i)
        summary = child.summary
        if summary is None:
            summary = child.value
            if summary is None:
                summary = "{...}"
        output.append(summary)

    return "(" + ", ".join(output) + ")"


class StdVecSyntheticProvider:
    """Pretty-printer for alloc::vec::Vec<T>

    struct Vec<T> { buf: RawVec<T>, len: usize }
    rust 1.75: struct RawVec<T> { ptr: Unique<T>, cap: usize, ... }
    rust 1.76: struct RawVec<T> { ptr: Unique<T>, cap: Cap(usize), ... }
    rust 1.31.1: struct Unique<T: ?Sized> { pointer: NonZero<*const T>, ... }
    rust 1.33.0: struct Unique<T: ?Sized> { pointer: *const T, ... }
    rust 1.62.0: struct Unique<T: ?Sized> { pointer: NonNull<T>, ... }
    struct NonZero<T>(T)
    struct NonNull<T> { pointer: *const T }
    """

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        # logger = Logger.Logger()
        # logger >> "[StdVecSyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj
        self.update()

    def num_children(self) -> int:
        return self.length

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + index * self.element_type_size
        element = self.data_ptr.CreateValueFromAddress(
            "[%s]" % index, address, self.element_type
        )
        return element

    def update(self):
        self.length = self.valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        self.buf = self.valobj.GetChildMemberWithName("buf").GetChildMemberWithName(
            "inner"
        )

        self.data_ptr = unwrap_unique_or_non_null(
            self.buf.GetChildMemberWithName("ptr")
        )

        self.element_type = self.valobj.GetType().GetTemplateArgumentType(0)

        if not self.element_type.IsValid():
            element_name = get_template_args(self.valobj.GetTypeName())[0]
            self.element_type = self.valobj.target.FindFirstType(element_name)

        self.element_type_size = self.element_type.GetByteSize()

    def has_children(self) -> bool:
        return True


class StdSliceSyntheticProvider:
    __slots__ = ["valobj", "length", "data_ptr", "element_type", "element_size"]

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj
        self.update()

    def num_children(self) -> int:
        return self.length

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + index * self.element_size
        element = self.data_ptr.CreateValueFromAddress(
            "[%s]" % index, address, self.element_type
        )
        return element

    def update(self):
        self.length = self.valobj.GetChildMemberWithName("length").GetValueAsUnsigned()
        self.data_ptr = self.valobj.GetChildMemberWithName("data_ptr")

        self.element_type = self.data_ptr.GetType().GetPointeeType()
        self.element_size = self.element_type.GetByteSize()

    def has_children(self) -> bool:
        return True


class MSVCStdSliceSyntheticProvider(StdSliceSyntheticProvider):
    def get_type_name(self) -> str:
        name = self.valobj.GetTypeName()

        if name.startswith("ref_mut"):
            # remove "ref_mut$<slice2$<" and trailing "> >"
            name = name[17:-3]
            ref = "&mut "
        else:
            # remove "ref$<slice2$<" and trailing "> >"
            name = name[13:-3]
            ref = "&"

        return "".join([ref, "[", name, "]"])


def StdSliceSummaryProvider(valobj, dict):
    output = sequence_formatter("[", valobj, dict)
    output += "]"
    return output


class StdVecDequeSyntheticProvider:
    """Pretty-printer for alloc::collections::vec_deque::VecDeque<T>

    struct VecDeque<T> { head: usize, len: usize, buf: RawVec<T> }
    """

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        # logger = Logger.Logger()
        # logger >> "[StdVecDequeSyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj
        self.update()

    def num_children(self) -> int:
        return self.size

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit() and int(index) < self.size:
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + ((index + self.head) % self.cap) * self.element_type_size
        element = self.data_ptr.CreateValueFromAddress(
            "[%s]" % index, address, self.element_type
        )
        return element

    def update(self):
        self.head = self.valobj.GetChildMemberWithName("head").GetValueAsUnsigned()
        self.size = self.valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        self.buf = self.valobj.GetChildMemberWithName("buf").GetChildMemberWithName(
            "inner"
        )
        cap = self.buf.GetChildMemberWithName("cap")
        if cap.GetType().num_fields == 1:
            cap = cap.GetChildAtIndex(0)
        self.cap = cap.GetValueAsUnsigned()

        self.data_ptr = unwrap_unique_or_non_null(
            self.buf.GetChildMemberWithName("ptr")
        )

        self.element_type = self.valobj.GetType().GetTemplateArgumentType(0)
        self.element_type_size = self.element_type.GetByteSize()

    def has_children(self) -> bool:
        return True


# BACKCOMPAT: rust 1.35
class StdOldHashMapSyntheticProvider:
    """Pretty-printer for std::collections::hash::map::HashMap<K, V, S>

    struct HashMap<K, V, S> {..., table: RawTable<K, V>, ... }
    struct RawTable<K, V> { capacity_mask: usize, size: usize, hashes: TaggedHashUintPtr, ... }
    """

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque, show_values: bool = True):
        self.valobj = valobj
        self.show_values = show_values
        self.update()

    def num_children(self) -> int:
        return self.size

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        # logger = Logger.Logger()
        start = self.data_ptr.GetValueAsUnsigned() & ~1

        # See `libstd/collections/hash/table.rs:raw_bucket_at
        hashes = self.hash_uint_size * self.capacity
        align = self.pair_type_size
        # See `libcore/alloc.rs:padding_needed_for`
        len_rounded_up = (
            (
                (((hashes + align) % self.modulo - 1) % self.modulo)
                & ~((align - 1) % self.modulo)
            )
            % self.modulo
            - hashes
        ) % self.modulo
        # len_rounded_up = ((hashes + align - 1) & ~(align - 1)) - hashes

        pairs_offset = hashes + len_rounded_up
        pairs_start = start + pairs_offset

        table_index = self.valid_indices[index]
        idx = table_index & self.capacity_mask
        address = pairs_start + idx * self.pair_type_size
        element = self.data_ptr.CreateValueFromAddress(
            "[%s]" % index, address, self.pair_type
        )
        if self.show_values:
            return element
        else:
            key = element.GetChildAtIndex(0)
            return self.valobj.CreateValueFromData(
                "[%s]" % index, key.GetData(), key.GetType()
            )

    def update(self):
        # logger = Logger.Logger()

        self.table = self.valobj.GetChildMemberWithName("table")  # type: SBValue
        self.size = self.table.GetChildMemberWithName("size").GetValueAsUnsigned()
        self.hashes = self.table.GetChildMemberWithName("hashes")
        self.hash_uint_type = self.hashes.GetType()
        self.hash_uint_size = self.hashes.GetType().GetByteSize()
        self.modulo = 2**self.hash_uint_size
        self.data_ptr = self.hashes.GetChildAtIndex(0).GetChildAtIndex(0)

        self.capacity_mask = self.table.GetChildMemberWithName(
            "capacity_mask"
        ).GetValueAsUnsigned()
        self.capacity = (self.capacity_mask + 1) % self.modulo

        marker = self.table.GetChildMemberWithName("marker").GetType()  # type: SBType
        self.pair_type = marker.template_args[0]
        self.pair_type_size = self.pair_type.GetByteSize()

        self.valid_indices = []
        for idx in range(self.capacity):
            address = self.data_ptr.GetValueAsUnsigned() + idx * self.hash_uint_size
            hash_uint = self.data_ptr.CreateValueFromAddress(
                "[%s]" % idx, address, self.hash_uint_type
            )
            hash_ptr = hash_uint.GetChildAtIndex(0).GetChildAtIndex(0)
            if hash_ptr.GetValueAsUnsigned() != 0:
                self.valid_indices.append(idx)

        # logger >> "Valid indices: {}".format(str(self.valid_indices))

    def has_children(self) -> bool:
        return True


class StdHashMapSyntheticProvider:
    """Pretty-printer for hashbrown's HashMap"""

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque, show_values: bool = True):
        self.valobj = valobj
        self.show_values = show_values
        self.update()

    def num_children(self) -> int:
        return self.size

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        pairs_start = self.data_ptr.GetValueAsUnsigned()
        idx = self.valid_indices[index]
        if self.new_layout:
            idx = -(idx + 1)
        address = pairs_start + idx * self.pair_type_size
        element = self.data_ptr.CreateValueFromAddress(
            "[%s]" % index, address, self.pair_type
        )
        if self.show_values:
            return element
        else:
            key = element.GetChildAtIndex(0)
            return self.valobj.CreateValueFromData(
                "[%s]" % index, key.GetData(), key.GetType()
            )

    def update(self):
        table = self.table()
        inner_table = table.GetChildMemberWithName("table")

        capacity = (
            inner_table.GetChildMemberWithName("bucket_mask").GetValueAsUnsigned() + 1
        )
        ctrl = inner_table.GetChildMemberWithName("ctrl").GetChildAtIndex(0)

        self.size = inner_table.GetChildMemberWithName("items").GetValueAsUnsigned()

        template_args = table.type.template_args

        if template_args is None:
            type_name = table.GetTypeName()
            args = get_template_args(type_name)
            self.pair_type = self.valobj.target.FindFirstType(args[0])
        else:
            self.pair_type = template_args[0]

        if self.pair_type.IsTypedefType():
            self.pair_type = self.pair_type.GetTypedefedType()
        self.pair_type_size = self.pair_type.GetByteSize()

        self.new_layout = not inner_table.GetChildMemberWithName("data").IsValid()
        if self.new_layout:
            self.data_ptr = ctrl.Cast(self.pair_type.GetPointerType())
        else:
            self.data_ptr = inner_table.GetChildMemberWithName("data").GetChildAtIndex(
                0
            )

        u8_type = self.valobj.GetTarget().GetBasicType(eBasicTypeUnsignedChar)
        u8_type_size = (
            self.valobj.GetTarget().GetBasicType(eBasicTypeUnsignedChar).GetByteSize()
        )

        self.valid_indices = []
        for idx in range(capacity):
            address = ctrl.GetValueAsUnsigned() + idx * u8_type_size
            value = ctrl.CreateValueFromAddress(
                "ctrl[%s]" % idx, address, u8_type
            ).GetValueAsUnsigned()
            is_present = value & 128 == 0
            if is_present:
                self.valid_indices.append(idx)

    def table(self) -> SBValue:
        if self.show_values:
            hashbrown_hashmap = self.valobj.GetChildMemberWithName("base")
        else:
            # BACKCOMPAT: rust 1.47
            # HashSet wraps either std HashMap or hashbrown::HashSet, which both
            # wrap hashbrown::HashMap, so either way we "unwrap" twice.
            hashbrown_hashmap = self.valobj.GetChildAtIndex(0).GetChildAtIndex(0)
        return hashbrown_hashmap.GetChildMemberWithName("table")

    def has_children(self) -> bool:
        return True


def StdRcSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    strong = valobj.GetChildMemberWithName("strong").GetValueAsUnsigned()
    weak = valobj.GetChildMemberWithName("weak").GetValueAsUnsigned()
    return "strong={}, weak={}".format(strong, weak)


class StdRcSyntheticProvider:
    """Pretty-printer for alloc::rc::Rc<T> and alloc::sync::Arc<T>

    struct Rc<T> { ptr: NonNull<RcInner<T>>, ... }
    rust 1.31.1: struct NonNull<T> { pointer: NonZero<*const T> }
    rust 1.33.0: struct NonNull<T> { pointer: *const T }
    struct NonZero<T>(T)
    struct RcInner<T> { strong: Cell<usize>, weak: Cell<usize>, value: T }
    struct Cell<T> { value: UnsafeCell<T> }
    struct UnsafeCell<T> { value: T }

    struct Arc<T> { ptr: NonNull<ArcInner<T>>, ... }
    struct ArcInner<T> { strong: atomic::AtomicUsize, weak: atomic::AtomicUsize, data: T }
    struct AtomicUsize { v: UnsafeCell<usize> }
    """

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque, is_atomic: bool = False):
        self.valobj = valobj

        self.ptr = unwrap_unique_or_non_null(self.valobj.GetChildMemberWithName("ptr"))

        self.value = self.ptr.GetChildMemberWithName("data" if is_atomic else "value")

        self.strong = (
            self.ptr.GetChildMemberWithName("strong")
            .GetChildAtIndex(0)
            .GetChildMemberWithName("value")
        )
        self.weak = (
            self.ptr.GetChildMemberWithName("weak")
            .GetChildAtIndex(0)
            .GetChildMemberWithName("value")
        )

        self.value_builder = ValueBuilder(valobj)

        self.update()

    def num_children(self) -> int:
        # Actually there are 3 children, but only the `value` should be shown as a child
        return 1

    def get_child_index(self, name: str) -> int:
        if name == "value":
            return 0
        if name == "strong":
            return 1
        if name == "weak":
            return 2
        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if index == 0:
            return self.value
        if index == 1:
            return self.value_builder.from_uint("strong", self.strong_count)
        if index == 2:
            return self.value_builder.from_uint("weak", self.weak_count)

        return None

    def update(self):
        self.strong_count = self.strong.GetValueAsUnsigned()
        self.weak_count = self.weak.GetValueAsUnsigned() - 1

    def has_children(self) -> bool:
        return True


class StdCellSyntheticProvider:
    """Pretty-printer for std::cell::Cell"""

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque):
        self.valobj = valobj
        self.value = valobj.GetChildMemberWithName("value").GetChildAtIndex(0)

    def num_children(self) -> int:
        return 1

    def get_child_index(self, name: str) -> int:
        if name == "value":
            return 0
        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if index == 0:
            return self.value
        return None

    def update(self):
        pass

    def has_children(self) -> bool:
        return True


def StdRefSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    borrow = valobj.GetChildMemberWithName("borrow").GetValueAsSigned()
    return (
        "borrow={}".format(borrow) if borrow >= 0 else "borrow_mut={}".format(-borrow)
    )


class StdRefSyntheticProvider:
    """Pretty-printer for std::cell::Ref, std::cell::RefMut, and std::cell::RefCell"""

    def __init__(self, valobj: SBValue, _dict: LLDBOpaque, is_cell: bool = False):
        self.valobj = valobj

        borrow = valobj.GetChildMemberWithName("borrow")
        value = valobj.GetChildMemberWithName("value")
        if is_cell:
            self.borrow = borrow.GetChildMemberWithName("value").GetChildMemberWithName(
                "value"
            )
            self.value = value.GetChildMemberWithName("value")
        else:
            self.borrow = (
                borrow.GetChildMemberWithName("borrow")
                .GetChildMemberWithName("value")
                .GetChildMemberWithName("value")
            )
            self.value = value.Dereference()

        self.value_builder = ValueBuilder(valobj)

        self.update()

    def num_children(self) -> int:
        # Actually there are 2 children, but only the `value` should be shown as a child
        return 1

    def get_child_index(self, name: str) -> int:
        if name == "value":
            return 0
        if name == "borrow":
            return 1
        return -1

    def get_child_at_index(self, index: int) -> SBValue:
        if index == 0:
            return self.value
        if index == 1:
            return self.value_builder.from_int("borrow", self.borrow_count)
        return None

    def update(self):
        self.borrow_count = self.borrow.GetValueAsSigned()

    def has_children(self) -> bool:
        return True


def StdNonZeroNumberSummaryProvider(valobj: SBValue, _dict: LLDBOpaque) -> str:
    inner = valobj.GetChildAtIndex(0)
    inner_inner = inner.GetChildAtIndex(0)

    # FIXME: Avoid printing as character literal,
    #        see https://github.com/llvm/llvm-project/issues/65076.
    if inner_inner.GetTypeName() in ["char", "unsigned char"]:
        return str(inner_inner.GetValueAsSigned())
    else:
        return inner_inner.GetValue()
