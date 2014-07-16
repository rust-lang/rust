# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import lldb

def print_val(val, internal_dict):
  '''Prints the given value with Rust syntax'''
  type_class = val.GetType().GetTypeClass()

  if type_class == lldb.eTypeClassStruct:
    return print_struct_val(val, internal_dict)

  if type_class == lldb.eTypeClassUnion:
    return print_enum_val(val, internal_dict)

  if type_class == lldb.eTypeClassPointer:
    return print_pointer_val(val, internal_dict)

  if type_class == lldb.eTypeClassArray:
    return print_fixed_size_vec_val(val, internal_dict)

  return val.GetValue()


#=--------------------------------------------------------------------------------------------------
# Type-Specialized Printing Functions
#=--------------------------------------------------------------------------------------------------

def print_struct_val(val, internal_dict):
  '''Prints a struct, tuple, or tuple struct value with Rust syntax'''
  assert val.GetType().GetTypeClass() == lldb.eTypeClassStruct

  if is_vec_slice(val):
    return print_vec_slice_val(val, internal_dict)
  else:
    return print_struct_val_starting_from(0, val, internal_dict)

def print_vec_slice_val(val, internal_dict):
  output = "&["

  length = val.GetChildAtIndex(1).GetValueAsUnsigned()

  data_ptr_val = val.GetChildAtIndex(0)
  data_ptr_type = data_ptr_val.GetType()
  assert data_ptr_type.IsPointerType()

  element_type = data_ptr_type.GetPointeeType()
  element_type_size = element_type.GetByteSize()

  start_address = data_ptr_val.GetValueAsUnsigned()

  for i in range(length):
    address = start_address + i * element_type_size
    element_val = val.CreateValueFromAddress( val.GetName() + ("[%s]" % i), address, element_type )
    output += print_val(element_val, internal_dict)

    if i != length - 1:
      output += ", "

  output += "]"
  return output

def print_struct_val_starting_from(field_start_index, val, internal_dict):
  '''
  Prints a struct, tuple, or tuple struct value with Rust syntax.
  Ignores any fields before field_start_index.
  '''
  assert val.GetType().GetTypeClass() == lldb.eTypeClassStruct

  t = val.GetType()
  has_field_names = type_has_field_names(t)
  type_name = extract_type_name(t.GetName())
  output = ""

  if not type_name.startswith("("):
    # this is a tuple, so don't print the type name
    output += type_name

  if has_field_names:
    output += " { \n"
  else:
    output += "("

  num_children = val.num_children

  for child_index in range(field_start_index, num_children):
    if has_field_names:
      field_name = t.GetFieldAtIndex(child_index).GetName()
      output += field_name + ": "

    field_val = val.GetChildAtIndex(child_index)
    output += print_val(field_val, internal_dict)

    if child_index != num_children - 1:
      output += ", "

    if has_field_names:
      output += "\n"

  if has_field_names:
    output += "}"
  else:
    output += ")"

  return output


def print_enum_val(val, internal_dict):
  '''Prints an enum value with Rust syntax'''

  assert val.GetType().GetTypeClass() == lldb.eTypeClassUnion

  if val.num_children == 1:
    first_variant_name = val.GetChildAtIndex(0).GetName()
    if first_variant_name and first_variant_name.startswith("RUST$ENCODED$ENUM$"):
      # Try to extract the

      last_separator_index = first_variant_name.rfind("$")
      if last_separator_index == -1:
        return "<invalid enum encoding: %s>" % first_variant_name

      second_last_separator_index = first_variant_name.rfind("$", 0, last_separator_index)
      if second_last_separator_index == -1:
        return "<invalid enum encoding: %s>" % first_variant_name

      try:
        disr_field_index = first_variant_name[second_last_separator_index + 1 :
                                              last_separator_index]
        disr_field_index = int(disr_field_index)
      except:
        return "<invalid enum encoding: %s>" % first_variant_name

      disr_val = val.GetChildAtIndex(0).GetChildAtIndex(disr_field_index).GetValueAsUnsigned()

      if disr_val == 0:
        null_variant_name = first_variant_name[last_separator_index + 1:]
        return null_variant_name
      else:
        return print_struct_val_starting_from(0, val.GetChildAtIndex(0), internal_dict)
    else:
      return print_struct_val_starting_from(0, val.GetChildAtIndex(0), internal_dict)

  # extract the discriminator value by
  disr_val = val.GetChildAtIndex(0).GetChildAtIndex(0)
  disr_type = disr_val.GetType()

  if disr_type.GetTypeClass() != lldb.eTypeClassEnumeration:
    return "<Invalid enum value encountered: Discriminator is not an enum>"

  variant_index = disr_val.GetValueAsUnsigned()
  return print_struct_val_starting_from(1, val.GetChildAtIndex(variant_index), internal_dict)


def print_pointer_val(val, internal_dict):
  '''Prints a pointer value with Rust syntax'''
  assert val.GetType().IsPointerType()
  sigil = "&"
  type_name = extract_type_name(val.GetType().GetName())
  if type_name and type_name[0:1] in ["&", "~", "*"]:
    sigil = type_name[0:1]

  return sigil + hex(val.GetValueAsUnsigned()) #print_val(val.Dereference(), internal_dict)


def print_fixed_size_vec_val(val, internal_dict):
  assert val.GetType().GetTypeClass() == lldb.eTypeClassArray

  output = "["

  for i in range(val.num_children):
    output += print_val(val.GetChildAtIndex(i), internal_dict)
    if i != val.num_children - 1:
      output += ", "

  output += "]"
  return output


#=--------------------------------------------------------------------------------------------------
# Helper Functions
#=--------------------------------------------------------------------------------------------------

unqualified_type_markers = frozenset(["(", "[", "&", "*"])

def extract_type_name(qualified_type_name):
  '''Extracts the type name from a fully qualified path'''
  if qualified_type_name[0] in unqualified_type_markers:
    return qualified_type_name

  end_of_search = qualified_type_name.find("<")
  if end_of_search < 0:
    end_of_search = len(qualified_type_name)

  index = qualified_type_name.rfind("::", 0, end_of_search)
  if index < 0:
    return qualified_type_name
  else:
    return qualified_type_name[index + 2:]


def type_has_field_names(ty):
  '''Returns true of this is a type with field names (struct, struct-like enum variant)'''
  # This may also be an enum variant where the first field doesn't have a name but the rest has
  if ty.GetNumberOfFields() > 1:
    return ty.GetFieldAtIndex(1).GetName() != None
  else:
    return ty.GetFieldAtIndex(0).GetName() != None


def is_vec_slice(val):
  ty = val.GetType()
  if ty.GetTypeClass() != lldb.eTypeClassStruct:
    return False

  if ty.GetNumberOfFields() != 2:
    return False

  if ty.GetFieldAtIndex(0).GetName() != "data_ptr":
    return False

  if ty.GetFieldAtIndex(1).GetName() != "length":
    return False

  type_name = extract_type_name(ty.GetName()).replace("&'static", "&").replace(" ", "")
  return type_name.startswith("&[") and type_name.endswith("]")
