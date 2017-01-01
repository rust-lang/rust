# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

"""
This module provides an abstraction layer over common Rust pretty printing
functionality needed by both GDB and LLDB.
"""

import re

# Type codes that indicate the kind of type as it appears in DWARF debug
# information. This code alone is not sufficient to determine the Rust type.
# For example structs, tuples, fat pointers, or enum variants will all have
# DWARF_TYPE_CODE_STRUCT.
DWARF_TYPE_CODE_STRUCT = 1
DWARF_TYPE_CODE_UNION  = 2
DWARF_TYPE_CODE_PTR    = 3
DWARF_TYPE_CODE_ARRAY  = 4
DWARF_TYPE_CODE_ENUM   = 5

# These constants specify the most specific kind of type that could be
# determined for a given value.
TYPE_KIND_UNKNOWN           = -1
TYPE_KIND_EMPTY             = 0
TYPE_KIND_SLICE             = 1
TYPE_KIND_REGULAR_STRUCT    = 2
TYPE_KIND_TUPLE             = 3
TYPE_KIND_TUPLE_STRUCT      = 4
TYPE_KIND_CSTYLE_VARIANT    = 5
TYPE_KIND_TUPLE_VARIANT     = 6
TYPE_KIND_STRUCT_VARIANT    = 7
TYPE_KIND_STR_SLICE         = 8
TYPE_KIND_STD_VEC           = 9
TYPE_KIND_STD_STRING        = 10
TYPE_KIND_REGULAR_ENUM      = 11
TYPE_KIND_COMPRESSED_ENUM   = 12
TYPE_KIND_SINGLETON_ENUM    = 13
TYPE_KIND_CSTYLE_ENUM       = 14
TYPE_KIND_PTR               = 15
TYPE_KIND_FIXED_SIZE_VEC    = 16
TYPE_KIND_REGULAR_UNION     = 17

ENCODED_ENUM_PREFIX = "RUST$ENCODED$ENUM$"
ENUM_DISR_FIELD_NAME = "RUST$ENUM$DISR"

# Slice related constants
SLICE_FIELD_NAME_DATA_PTR = "data_ptr"
SLICE_FIELD_NAME_LENGTH = "length"
SLICE_FIELD_NAMES = [SLICE_FIELD_NAME_DATA_PTR, SLICE_FIELD_NAME_LENGTH]

# std::Vec<> related constants
STD_VEC_FIELD_NAME_LENGTH = "len"
STD_VEC_FIELD_NAME_BUF = "buf"
STD_VEC_FIELD_NAMES = [STD_VEC_FIELD_NAME_BUF,
                       STD_VEC_FIELD_NAME_LENGTH]

# std::String related constants
STD_STRING_FIELD_NAMES = ["vec"]


class Type(object):
    """
    This class provides a common interface for type-oriented operations.
    Sub-classes are supposed to wrap a debugger-specific type-object and
    provide implementations for the abstract methods in this class.
    """

    def __init__(self):
        self.__type_kind = None

    def get_unqualified_type_name(self):
        """
        Implementations of this method should return the unqualified name of the
        type-object they are wrapping. Some examples:

        'int' -> 'int'
        'std::vec::Vec<std::string::String>' -> 'Vec<std::string::String>'
        '&std::option::Option<std::string::String>' -> '&std::option::Option<std::string::String>'

        As you can see, type arguments stay fully qualified.
        """
        raise NotImplementedError("Override this method")

    def get_dwarf_type_kind(self):
        """
        Implementations of this method should return the correct
        DWARF_TYPE_CODE_* value for the wrapped type-object.
        """
        raise NotImplementedError("Override this method")

    def get_fields(self):
        """
        Implementations of this method should return a list of field-objects of
        this type. For Rust-enums (i.e. with DWARF_TYPE_CODE_UNION) these field-
        objects represent the variants of the enum. Field-objects must have a
        `name` attribute that gives their name as specified in DWARF.
        """
        assert ((self.get_dwarf_type_kind() == DWARF_TYPE_CODE_STRUCT) or
                (self.get_dwarf_type_kind() == DWARF_TYPE_CODE_UNION))
        raise NotImplementedError("Override this method")

    def get_wrapped_value(self):
        """
        Returns the debugger-specific type-object wrapped by this object. This
        is sometimes needed for doing things like pointer-arithmetic in GDB.
        """
        raise NotImplementedError("Override this method")

    def get_type_kind(self):
        """This method returns the TYPE_KIND_* value for this type-object."""
        if self.__type_kind is None:
            dwarf_type_code = self.get_dwarf_type_kind()

            if dwarf_type_code == DWARF_TYPE_CODE_STRUCT:
                self.__type_kind = self.__classify_struct()
            elif dwarf_type_code == DWARF_TYPE_CODE_UNION:
                self.__type_kind = self.__classify_union()
            elif dwarf_type_code == DWARF_TYPE_CODE_PTR:
                self.__type_kind = TYPE_KIND_PTR
            elif dwarf_type_code == DWARF_TYPE_CODE_ARRAY:
                self.__type_kind = TYPE_KIND_FIXED_SIZE_VEC
            else:
                self.__type_kind = TYPE_KIND_UNKNOWN
        return self.__type_kind

    def __classify_struct(self):
        assert self.get_dwarf_type_kind() == DWARF_TYPE_CODE_STRUCT

        unqualified_type_name = self.get_unqualified_type_name()

        # STR SLICE
        if unqualified_type_name == "&str":
            return TYPE_KIND_STR_SLICE

        # REGULAR SLICE
        if (unqualified_type_name.startswith(("&[", "&mut [")) and
            unqualified_type_name.endswith("]") and
            self.__conforms_to_field_layout(SLICE_FIELD_NAMES)):
            return TYPE_KIND_SLICE

        fields = self.get_fields()
        field_count = len(fields)

        # EMPTY STRUCT
        if field_count == 0:
            return TYPE_KIND_EMPTY

        # STD VEC
        if (unqualified_type_name.startswith("Vec<") and
            self.__conforms_to_field_layout(STD_VEC_FIELD_NAMES)):
            return TYPE_KIND_STD_VEC

        # STD STRING
        if (unqualified_type_name.startswith("String") and
            self.__conforms_to_field_layout(STD_STRING_FIELD_NAMES)):
            return TYPE_KIND_STD_STRING

        # ENUM VARIANTS
        if fields[0].name == ENUM_DISR_FIELD_NAME:
            if field_count == 1:
                return TYPE_KIND_CSTYLE_VARIANT
            elif self.__all_fields_conform_to_tuple_field_naming(1):
                return TYPE_KIND_TUPLE_VARIANT
            else:
                return TYPE_KIND_STRUCT_VARIANT

        # TUPLE
        if self.__all_fields_conform_to_tuple_field_naming(0):
            if unqualified_type_name.startswith("("):
                return TYPE_KIND_TUPLE
            else:
                return TYPE_KIND_TUPLE_STRUCT

        # REGULAR STRUCT
        return TYPE_KIND_REGULAR_STRUCT


    def __classify_union(self):
        assert self.get_dwarf_type_kind() == DWARF_TYPE_CODE_UNION

        union_members = self.get_fields()
        union_member_count = len(union_members)
        if union_member_count == 0:
            return TYPE_KIND_EMPTY

        first_variant_name = union_members[0].name
        if first_variant_name is None:
            if union_member_count == 1:
                return TYPE_KIND_SINGLETON_ENUM
            else:
                return TYPE_KIND_REGULAR_ENUM
        elif first_variant_name.startswith(ENCODED_ENUM_PREFIX):
            assert union_member_count == 1
            return TYPE_KIND_COMPRESSED_ENUM
        else:
            return TYPE_KIND_REGULAR_UNION


    def __conforms_to_field_layout(self, expected_fields):
        actual_fields = self.get_fields()
        actual_field_count = len(actual_fields)

        if actual_field_count != len(expected_fields):
            return False

        for i in range(0, actual_field_count):
            if actual_fields[i].name != expected_fields[i]:
                return False

        return True

    def __all_fields_conform_to_tuple_field_naming(self, start_index):
        fields = self.get_fields()
        field_count = len(fields)

        for i in range(start_index, field_count):
            field_name = fields[i].name
            if (field_name is None) or (re.match(r"__\d+$", field_name) is None):
                return False
        return True


class Value(object):
    """
    This class provides a common interface for value-oriented operations.
    Sub-classes are supposed to wrap a debugger-specific value-object and
    provide implementations for the abstract methods in this class.
    """
    def __init__(self, ty):
        self.type = ty

    def get_child_at_index(self, index):
        """Returns the value of the field, array element or variant at the given index"""
        raise NotImplementedError("Override this method")

    def as_integer(self):
        """
        Try to convert the wrapped value into a Python integer. This should
        always succeed for values that are pointers or actual integers.
        """
        raise NotImplementedError("Override this method")

    def get_wrapped_value(self):
        """
        Returns the debugger-specific value-object wrapped by this object. This
        is sometimes needed for doing things like pointer-arithmetic in GDB.
        """
        raise NotImplementedError("Override this method")


class EncodedEnumInfo(object):
    """
    This class provides facilities for handling enum values with compressed
    encoding where a non-null field in one variant doubles as the discriminant.
    """

    def __init__(self, enum_val):
        assert enum_val.type.get_type_kind() == TYPE_KIND_COMPRESSED_ENUM
        variant_name = enum_val.type.get_fields()[0].name
        last_separator_index = variant_name.rfind("$")
        start_index = len(ENCODED_ENUM_PREFIX)
        indices_substring = variant_name[start_index:last_separator_index].split("$")
        self.__enum_val = enum_val
        self.__disr_field_indices = [int(index) for index in indices_substring]
        self.__null_variant_name = variant_name[last_separator_index + 1:]

    def is_null_variant(self):
        ty = self.__enum_val.type
        sole_variant_val = self.__enum_val.get_child_at_index(0)
        discriminant_val = sole_variant_val
        for disr_field_index in self.__disr_field_indices:
            discriminant_val = discriminant_val.get_child_at_index(disr_field_index)

        # If the discriminant field is a fat pointer we have to consider the
        # first word as the true discriminant
        if discriminant_val.type.get_dwarf_type_kind() == DWARF_TYPE_CODE_STRUCT:
            discriminant_val = discriminant_val.get_child_at_index(0)

        return discriminant_val.as_integer() == 0

    def get_non_null_variant_val(self):
        return self.__enum_val.get_child_at_index(0)

    def get_null_variant_name(self):
        return self.__null_variant_name


def get_discriminant_value_as_integer(enum_val):
    assert enum_val.type.get_dwarf_type_kind() == DWARF_TYPE_CODE_UNION
    # we can take any variant here because the discriminant has to be the same
    # for all of them.
    variant_val = enum_val.get_child_at_index(0)
    disr_val = variant_val.get_child_at_index(0)
    return disr_val.as_integer()


def extract_length_ptr_and_cap_from_std_vec(vec_val):
    assert vec_val.type.get_type_kind() == TYPE_KIND_STD_VEC
    length_field_index = STD_VEC_FIELD_NAMES.index(STD_VEC_FIELD_NAME_LENGTH)
    buf_field_index = STD_VEC_FIELD_NAMES.index(STD_VEC_FIELD_NAME_BUF)

    length = vec_val.get_child_at_index(length_field_index).as_integer()
    buf = vec_val.get_child_at_index(buf_field_index)

    vec_ptr_val = buf.get_child_at_index(0)
    capacity = buf.get_child_at_index(1).as_integer()
    unique_ptr_val = vec_ptr_val.get_child_at_index(0)
    data_ptr = unique_ptr_val.get_child_at_index(0)
    assert data_ptr.type.get_dwarf_type_kind() == DWARF_TYPE_CODE_PTR
    return (length, data_ptr, capacity)

def extract_length_and_ptr_from_slice(slice_val):
    assert (slice_val.type.get_type_kind() == TYPE_KIND_SLICE or
            slice_val.type.get_type_kind() == TYPE_KIND_STR_SLICE)

    length_field_index = SLICE_FIELD_NAMES.index(SLICE_FIELD_NAME_LENGTH)
    ptr_field_index = SLICE_FIELD_NAMES.index(SLICE_FIELD_NAME_DATA_PTR)

    length = slice_val.get_child_at_index(length_field_index).as_integer()
    data_ptr = slice_val.get_child_at_index(ptr_field_index)

    assert data_ptr.type.get_dwarf_type_kind() == DWARF_TYPE_CODE_PTR
    return (length, data_ptr)

UNQUALIFIED_TYPE_MARKERS = frozenset(["(", "[", "&", "*"])

def extract_type_name(qualified_type_name):
    """Extracts the type name from a fully qualified path"""
    if qualified_type_name[0] in UNQUALIFIED_TYPE_MARKERS:
        return qualified_type_name

    end_of_search = qualified_type_name.find("<")
    if end_of_search < 0:
        end_of_search = len(qualified_type_name)

    index = qualified_type_name.rfind("::", 0, end_of_search)
    if index < 0:
        return qualified_type_name
    else:
        return qualified_type_name[index + 2:]
