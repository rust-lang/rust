# Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import gdb

#===============================================================================
# GDB Pretty Printing Module for Rust
#===============================================================================

def register_printers(objfile):
  "Registers Rust pretty printers for the given objfile"
  objfile.pretty_printers.append(rust_pretty_printer_lookup_function)

def rust_pretty_printer_lookup_function(val):
  "Returns the correct Rust pretty printer for the given value if there is one"
  type_code = val.type.code

  if type_code == gdb.TYPE_CODE_STRUCT:
    struct_kind = classify_struct(val.type)

    if struct_kind == STRUCT_KIND_STR_SLICE:
      return RustStringSlicePrinter(val)

    if struct_kind == STRUCT_KIND_TUPLE:
      return RustTuplePrinter(val)

    if struct_kind == STRUCT_KIND_TUPLE_STRUCT:
      return RustTupleStructPrinter(val, False)

    if struct_kind == STRUCT_KIND_CSTYLE_VARIANT:
      return RustCStyleEnumPrinter(val[get_field_at_index(val, 0)])

    if struct_kind == STRUCT_KIND_TUPLE_VARIANT:
      return RustTupleStructPrinter(val, True)

    if struct_kind == STRUCT_KIND_STRUCT_VARIANT:
      return RustStructPrinter(val, True)

    return RustStructPrinter(val, False)

  # Enum handling
  if type_code == gdb.TYPE_CODE_UNION:
    enum_members = list(val.type.fields())
    enum_member_count = len(enum_members)

    if enum_member_count == 0:
      return RustStructPrinter(val, false)

    if enum_member_count == 1:
      if enum_members[0].name == None:
        # This is a singleton enum
        return rust_pretty_printer_lookup_function(val[enum_members[0]])
      else:
        assert enum_members[0].name.startswith("RUST$ENCODED$ENUM$")
        # This is a space-optimized enum
        last_separator_index = enum_members[0].name.rfind("$")
        second_last_separator_index = first_variant_name.rfind("$", 0, last_separator_index)
        disr_field_index = first_variant_name[second_last_separator_index + 1 :
                                              last_separator_index]
        disr_field_index = int(disr_field_index)

        sole_variant_val = val[enum_members[0]]
        disr_field = get_field_at_index(sole_variant_val, disr_field_index)
        discriminant = int(sole_variant_val[disr_field])

        if discriminant == 0:
          null_variant_name = first_variant_name[last_separator_index + 1:]
          return IdentityPrinter(null_variant_name)

        return rust_pretty_printer_lookup_function(sole_variant_val)

    # This is a regular enum, extract the discriminant
    discriminant_name, discriminant_val = extract_discriminant_value(val)
    return rust_pretty_printer_lookup_function(val[enum_members[discriminant_val]])

  # No pretty printer has been found
  return None

#=------------------------------------------------------------------------------
# Pretty Printer Classes
#=------------------------------------------------------------------------------

class RustStructPrinter:
  def __init__(self, val, hide_first_field):
    self.val = val
    self.hide_first_field = hide_first_field

  def to_string(self):
    return self.val.type.tag

  def children(self):
    cs = []
    for field in self.val.type.fields():
      field_name = field.name
      # Normally the field name is used as a key to access the field value,
      # because that's also supported in older versions of GDB...
      field_key = field_name
      if field_name == None:
        field_name = ""
        # ... but for fields without a name (as in tuples), we have to fall back
        # to the newer method of using the field object directly as key. In
        # older versions of GDB, this will just fail.
        field_key = field
      name_value_tuple = ( field_name, self.val[field_key] )
      cs.append( name_value_tuple )

    if self.hide_first_field:
      cs = cs[1:]

    return cs

class RustTuplePrinter:
  def __init__(self, val):
    self.val = val

  def to_string(self):
    return None

  def children(self):
    cs = []
    for field in self.val.type.fields():
      cs.append( ("", self.val[field]) )

    return cs

  def display_hint(self):
    return "array"

class RustTupleStructPrinter:
  def __init__(self, val, hide_first_field):
    self.val = val
    self.hide_first_field = hide_first_field

  def to_string(self):
    return self.val.type.tag

  def children(self):
    cs = []
    for field in self.val.type.fields():
      cs.append( ("", self.val[field]) )

    if self.hide_first_field:
      cs = cs[1:]

    return cs

  def display_hint(self):
    return "array"

class RustStringSlicePrinter:
  def __init__(self, val):
    self.val = val

  def to_string(self):
    slice_byte_len = self.val["length"]
    return '"%s"' % self.val["data_ptr"].string(encoding = "utf-8",
                                                length = slice_byte_len)

class RustCStyleEnumPrinter:
  def __init__(self, val):
    assert val.type.code == gdb.TYPE_CODE_ENUM
    self.val = val

  def to_string(self):
    return str(self.val)

class IdentityPrinter:
  def __init__(self, string):
    self.string

  def to_string(self):
    return self.string

STRUCT_KIND_REGULAR_STRUCT  = 0
STRUCT_KIND_TUPLE_STRUCT    = 1
STRUCT_KIND_TUPLE           = 2
STRUCT_KIND_TUPLE_VARIANT   = 3
STRUCT_KIND_STRUCT_VARIANT  = 4
STRUCT_KIND_CSTYLE_VARIANT  = 5
STRUCT_KIND_STR_SLICE       = 6

def classify_struct(type):
  if type.tag == "&str":
    return STRUCT_KIND_STR_SLICE

  fields = list(type.fields())
  field_count = len(fields)

  if field_count == 0:
    return STRUCT_KIND_REGULAR_STRUCT

  if fields[0].artificial:
    if field_count == 1:
      return STRUCT_KIND_CSTYLE_VARIANT
    elif fields[1].name == None:
      return STRUCT_KIND_TUPLE_VARIANT
    else:
      return STRUCT_KIND_STRUCT_VARIANT

  if fields[0].name == None:
    if type.tag.startswith("("):
      return STRUCT_KIND_TUPLE
    else:
      return STRUCT_KIND_TUPLE_STRUCT

  return STRUCT_KIND_REGULAR_STRUCT

def extract_discriminant_value(enum_val):
  assert enum_val.type.code == gdb.TYPE_CODE_UNION
  for variant_descriptor in enum_val.type.fields():
    variant_val = enum_val[variant_descriptor]
    for field in variant_val.type.fields():
      return (field.name, int(variant_val[field]))

def first_field(val):
  for field in val.type.fields():
    return field

def get_field_at_index(val, index):
  i = 0
  for field in val.type.fields():
    if i == index:
      return field
  return None
