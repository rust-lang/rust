(*
 * Walk crate and generate DWARF-3 records. This file might also go in
 * the be/ directory; it's half-middle-end, half-back-end. Debug info is
 * like that.
 *
 * Some notes about DWARF:
 *
 *   - Records form an ownership tree. The tree is serialized in
 *     depth-first pre-order with child lists ending with null
 *     records. When a node type is defined to have no children, no null
 *     child record is provided; it's implied.
 *
 *               [parent]
 *                /    \
 *          [child1]  [child2]
 *              |
 *          [grandchild1]
 *
 *     serializes as:
 *
 *          [parent][child1][grandchild1][null][child2][null][null]
 *
 *   - Sometimes you want to make it possible to scan through a sibling
 *     list quickly while skipping the sub-children of each (such as
 *     skipping the 'grandchild' above); this can be done with a
 *     DW_AT_sibling attribute that points forward to the next same-level
 *     sibling.
 *
 *   - A DWARF consumer contains a little stack-machine interpreter for
 *     a micro-language that you can embed in DWARF records to compute
 *     values algorithmically.
 *
 *   - DWARF is not "officially" supported by any Microsoft tools in
 *     PE files, but the Microsoft debugging information formats are
 *     proprietary and ever-shifting, and not clearly sufficient for
 *     our needs; by comparison DWARF is widely supported, stable,
 *     flexible, and required everywhere *else*. We are using DWARF to
 *     support major components of the rust runtime (reflection,
 *     unwinding, profiling) so it's helpful to not have to span
 *     technologies, just focus on DWARF.  Luckily the MINGW/Cygwin
 *     communities have worked out a convention for PE, and taught BFD
 *     (thus most tools) how to digest DWARF sections trailing after
 *     the .idata section of a normal PE file. Seems to work fine.
 * 
 *   - DWARF supports variable-length coding using LEB128, and in the
 *     cases where these are symbolic or self-contained numbers, we
 *     support them in the assembler. Inter-DWARF-record references
 *     can be done via fixed-size DW_FORM_ref{1,2,4,8} or
 *     DW_FORM_ref_addr; or else via variable-size (LEB128)
 *     DW_FORM_ref_udata. It is hazardous to use the LEB128 form in
 *     our implementation of references, since we use a generic 2-pass
 *     (+ relaxation) fixup mechanism in our assembler which in
 *     general may present an information-dependency cycle for LEB128
 *     coding of offsets: you need to know the offset before you can
 *     work out the LEB128 size, and you may need to know several
 *     LEB128-sizes before you can work out the offsets of other
 *     LEB128s (possibly even the one you're currently coding). In
 *     general the assembler makes no attempt to resolve such
 *     cycles. It'll just throw if it can't handle what you ask
 *     for. So it's best to pay a little extra space and use
 *     DW_FORM_ref_addr or DW_FORM_ref{1,2,4,8} values, in all cases.
 *)

open Semant;;
open Common;;
open Asm;;

let log cx = Session.log "dwarf"
  cx.ctxt_sess.Session.sess_log_dwarf
  cx.ctxt_sess.Session.sess_log_out
;;

type dw_tag =
    DW_TAG_array_type
  | DW_TAG_class_type
  | DW_TAG_entry_point
  | DW_TAG_enumeration_type
  | DW_TAG_formal_parameter
  | DW_TAG_imported_declaration
  | DW_TAG_label
  | DW_TAG_lexical_block
  | DW_TAG_member
  | DW_TAG_pointer_type
  | DW_TAG_reference_type
  | DW_TAG_compile_unit
  | DW_TAG_string_type
  | DW_TAG_structure_type
  | DW_TAG_subroutine_type
  | DW_TAG_typedef
  | DW_TAG_union_type
  | DW_TAG_unspecified_parameters
  | DW_TAG_variant
  | DW_TAG_common_block
  | DW_TAG_common_inclusion
  | DW_TAG_inheritance
  | DW_TAG_inlined_subroutine
  | DW_TAG_module
  | DW_TAG_ptr_to_member_type
  | DW_TAG_set_type
  | DW_TAG_subrange_type
  | DW_TAG_with_stmt
  | DW_TAG_access_declaration
  | DW_TAG_base_type
  | DW_TAG_catch_block
  | DW_TAG_const_type
  | DW_TAG_constant
  | DW_TAG_enumerator
  | DW_TAG_file_type
  | DW_TAG_friend
  | DW_TAG_namelist
  | DW_TAG_namelist_item
  | DW_TAG_packed_type
  | DW_TAG_subprogram
  | DW_TAG_template_type_parameter
  | DW_TAG_template_value_parameter
  | DW_TAG_thrown_type
  | DW_TAG_try_block
  | DW_TAG_variant_part
  | DW_TAG_variable
  | DW_TAG_volatile_type
  | DW_TAG_dwarf_procedure
  | DW_TAG_restrict_type
  | DW_TAG_interface_type
  | DW_TAG_namespace
  | DW_TAG_imported_module
  | DW_TAG_unspecified_type
  | DW_TAG_partial_unit
  | DW_TAG_imported_unit
  | DW_TAG_condition
  | DW_TAG_shared_type
  | DW_TAG_lo_user
  | DW_TAG_rust_meta
  | DW_TAG_hi_user
;;


let dw_tag_to_int (tag:dw_tag) : int =
  match tag with
    DW_TAG_array_type -> 0x01
  | DW_TAG_class_type -> 0x02
  | DW_TAG_entry_point -> 0x03
  | DW_TAG_enumeration_type -> 0x04
  | DW_TAG_formal_parameter -> 0x05
  | DW_TAG_imported_declaration -> 0x08
  | DW_TAG_label -> 0x0a
  | DW_TAG_lexical_block -> 0x0b
  | DW_TAG_member -> 0x0d
  | DW_TAG_pointer_type -> 0x0f
  | DW_TAG_reference_type -> 0x10
  | DW_TAG_compile_unit -> 0x11
  | DW_TAG_string_type -> 0x12
  | DW_TAG_structure_type -> 0x13
  | DW_TAG_subroutine_type -> 0x15
  | DW_TAG_typedef -> 0x16
  | DW_TAG_union_type -> 0x17
  | DW_TAG_unspecified_parameters -> 0x18
  | DW_TAG_variant -> 0x19
  | DW_TAG_common_block -> 0x1a
  | DW_TAG_common_inclusion -> 0x1b
  | DW_TAG_inheritance -> 0x1c
  | DW_TAG_inlined_subroutine -> 0x1d
  | DW_TAG_module -> 0x1e
  | DW_TAG_ptr_to_member_type -> 0x1f
  | DW_TAG_set_type -> 0x20
  | DW_TAG_subrange_type -> 0x21
  | DW_TAG_with_stmt -> 0x22
  | DW_TAG_access_declaration -> 0x23
  | DW_TAG_base_type -> 0x24
  | DW_TAG_catch_block -> 0x25
  | DW_TAG_const_type -> 0x26
  | DW_TAG_constant -> 0x27
  | DW_TAG_enumerator -> 0x28
  | DW_TAG_file_type -> 0x29
  | DW_TAG_friend -> 0x2a
  | DW_TAG_namelist -> 0x2b
  | DW_TAG_namelist_item -> 0x2c
  | DW_TAG_packed_type -> 0x2d
  | DW_TAG_subprogram -> 0x2e
  | DW_TAG_template_type_parameter -> 0x2f
  | DW_TAG_template_value_parameter -> 0x30
  | DW_TAG_thrown_type -> 0x31
  | DW_TAG_try_block -> 0x32
  | DW_TAG_variant_part -> 0x33
  | DW_TAG_variable -> 0x34
  | DW_TAG_volatile_type -> 0x35
  | DW_TAG_dwarf_procedure -> 0x36
  | DW_TAG_restrict_type -> 0x37
  | DW_TAG_interface_type -> 0x38
  | DW_TAG_namespace -> 0x39
  | DW_TAG_imported_module -> 0x3a
  | DW_TAG_unspecified_type -> 0x3b
  | DW_TAG_partial_unit -> 0x3c
  | DW_TAG_imported_unit -> 0x3d
  | DW_TAG_condition -> 0x3f
  | DW_TAG_shared_type -> 0x40
  | DW_TAG_lo_user -> 0x4080
  | DW_TAG_rust_meta -> 0x4300
  | DW_TAG_hi_user -> 0xffff
;;

let dw_tag_of_int (i:int) : dw_tag =
  match i with
    0x01 -> DW_TAG_array_type
  | 0x02 -> DW_TAG_class_type
  | 0x03 -> DW_TAG_entry_point
  | 0x04 -> DW_TAG_enumeration_type
  | 0x05 -> DW_TAG_formal_parameter
  | 0x08 -> DW_TAG_imported_declaration
  | 0x0a -> DW_TAG_label
  | 0x0b -> DW_TAG_lexical_block
  | 0x0d -> DW_TAG_member
  | 0x0f -> DW_TAG_pointer_type
  | 0x10 -> DW_TAG_reference_type
  | 0x11 -> DW_TAG_compile_unit
  | 0x12 -> DW_TAG_string_type
  | 0x13 -> DW_TAG_structure_type
  | 0x15 -> DW_TAG_subroutine_type
  | 0x16 -> DW_TAG_typedef
  | 0x17 -> DW_TAG_union_type
  | 0x18 -> DW_TAG_unspecified_parameters
  | 0x19 -> DW_TAG_variant
  | 0x1a -> DW_TAG_common_block
  | 0x1b -> DW_TAG_common_inclusion
  | 0x1c -> DW_TAG_inheritance
  | 0x1d -> DW_TAG_inlined_subroutine
  | 0x1e -> DW_TAG_module
  | 0x1f -> DW_TAG_ptr_to_member_type
  | 0x20 -> DW_TAG_set_type
  | 0x21 -> DW_TAG_subrange_type
  | 0x22 -> DW_TAG_with_stmt
  | 0x23 -> DW_TAG_access_declaration
  | 0x24 -> DW_TAG_base_type
  | 0x25 -> DW_TAG_catch_block
  | 0x26 -> DW_TAG_const_type
  | 0x27 -> DW_TAG_constant
  | 0x28 -> DW_TAG_enumerator
  | 0x29 -> DW_TAG_file_type
  | 0x2a -> DW_TAG_friend
  | 0x2b -> DW_TAG_namelist
  | 0x2c -> DW_TAG_namelist_item
  | 0x2d -> DW_TAG_packed_type
  | 0x2e -> DW_TAG_subprogram
  | 0x2f -> DW_TAG_template_type_parameter
  | 0x30 -> DW_TAG_template_value_parameter
  | 0x31 -> DW_TAG_thrown_type
  | 0x32 -> DW_TAG_try_block
  | 0x33 -> DW_TAG_variant_part
  | 0x34 -> DW_TAG_variable
  | 0x35 -> DW_TAG_volatile_type
  | 0x36 -> DW_TAG_dwarf_procedure
  | 0x37 -> DW_TAG_restrict_type
  | 0x38 -> DW_TAG_interface_type
  | 0x39 -> DW_TAG_namespace
  | 0x3a -> DW_TAG_imported_module
  | 0x3b -> DW_TAG_unspecified_type
  | 0x3c -> DW_TAG_partial_unit
  | 0x3d -> DW_TAG_imported_unit
  | 0x3f -> DW_TAG_condition
  | 0x40 -> DW_TAG_shared_type
  | 0x4080 -> DW_TAG_lo_user
  | 0x4300 -> DW_TAG_rust_meta
  | 0xffff -> DW_TAG_hi_user
  | _ -> bug () "bad DWARF tag code: %d" i
;;


let dw_tag_to_string (tag:dw_tag) : string =
  match tag with
    DW_TAG_array_type -> "DW_TAG_array_type"
  | DW_TAG_class_type -> "DW_TAG_class_type"
  | DW_TAG_entry_point -> "DW_TAG_entry_point"
  | DW_TAG_enumeration_type -> "DW_TAG_enumeration_type"
  | DW_TAG_formal_parameter -> "DW_TAG_formal_parameter"
  | DW_TAG_imported_declaration -> "DW_TAG_imported_declaration"
  | DW_TAG_label -> "DW_TAG_label"
  | DW_TAG_lexical_block -> "DW_TAG_lexical_block"
  | DW_TAG_member -> "DW_TAG_member"
  | DW_TAG_pointer_type -> "DW_TAG_pointer_type"
  | DW_TAG_reference_type -> "DW_TAG_reference_type"
  | DW_TAG_compile_unit -> "DW_TAG_compile_unit"
  | DW_TAG_string_type -> "DW_TAG_string_type"
  | DW_TAG_structure_type -> "DW_TAG_structure_type"
  | DW_TAG_subroutine_type -> "DW_TAG_subroutine_type"
  | DW_TAG_typedef -> "DW_TAG_typedef"
  | DW_TAG_union_type -> "DW_TAG_union_type"
  | DW_TAG_unspecified_parameters -> "DW_TAG_unspecified_parameters"
  | DW_TAG_variant -> "DW_TAG_variant"
  | DW_TAG_common_block -> "DW_TAG_common_block"
  | DW_TAG_common_inclusion -> "DW_TAG_common_inclusion"
  | DW_TAG_inheritance -> "DW_TAG_inheritance"
  | DW_TAG_inlined_subroutine -> "DW_TAG_inlined_subroutine"
  | DW_TAG_module -> "DW_TAG_module"
  | DW_TAG_ptr_to_member_type -> "DW_TAG_ptr_to_member_type"
  | DW_TAG_set_type -> "DW_TAG_set_type"
  | DW_TAG_subrange_type -> "DW_TAG_subrange_type"
  | DW_TAG_with_stmt -> "DW_TAG_with_stmt"
  | DW_TAG_access_declaration -> "DW_TAG_access_declaration"
  | DW_TAG_base_type -> "DW_TAG_base_type"
  | DW_TAG_catch_block -> "DW_TAG_catch_block"
  | DW_TAG_const_type -> "DW_TAG_const_type"
  | DW_TAG_constant -> "DW_TAG_constant"
  | DW_TAG_enumerator -> "DW_TAG_enumerator"
  | DW_TAG_file_type -> "DW_TAG_file_type"
  | DW_TAG_friend -> "DW_TAG_friend"
  | DW_TAG_namelist -> "DW_TAG_namelist"
  | DW_TAG_namelist_item -> "DW_TAG_namelist_item"
  | DW_TAG_packed_type -> "DW_TAG_packed_type"
  | DW_TAG_subprogram -> "DW_TAG_subprogram"
  | DW_TAG_template_type_parameter -> "DW_TAG_template_type_parameter"
  | DW_TAG_template_value_parameter -> "DW_TAG_template_value_parameter"
  | DW_TAG_thrown_type -> "DW_TAG_thrown_type"
  | DW_TAG_try_block -> "DW_TAG_try_block"
  | DW_TAG_variant_part -> "DW_TAG_variant_part"
  | DW_TAG_variable -> "DW_TAG_variable"
  | DW_TAG_volatile_type -> "DW_TAG_volatile_type"
  | DW_TAG_dwarf_procedure -> "DW_TAG_dwarf_procedure"
  | DW_TAG_restrict_type -> "DW_TAG_restrict_type"
  | DW_TAG_interface_type -> "DW_TAG_interface_type"
  | DW_TAG_namespace -> "DW_TAG_namespace"
  | DW_TAG_imported_module -> "DW_TAG_imported_module"
  | DW_TAG_unspecified_type -> "DW_TAG_unspecified_type"
  | DW_TAG_partial_unit -> "DW_TAG_partial_unit"
  | DW_TAG_imported_unit -> "DW_TAG_imported_unit"
  | DW_TAG_condition -> "DW_TAG_condition"
  | DW_TAG_shared_type -> "DW_TAG_shared_type"
  | DW_TAG_lo_user -> "DW_TAG_lo_user"
  | DW_TAG_rust_meta -> "DW_TAG_rust_meta"
  | DW_TAG_hi_user -> "DW_TAG_hi_user"
;;


type dw_children =
    DW_CHILDREN_no
  | DW_CHILDREN_yes
;;


let dw_children_to_int (ch:dw_children) : int =
  match ch with
      DW_CHILDREN_no -> 0x00
    | DW_CHILDREN_yes -> 0x01
;;

let dw_children_of_int (i:int) : dw_children =
  match i with
      0 -> DW_CHILDREN_no
    | 1 -> DW_CHILDREN_yes
    | _ -> bug () "bad DWARF children code: %d" i
;;

type dw_at =
    DW_AT_sibling
  | DW_AT_location
  | DW_AT_name
  | DW_AT_ordering
  | DW_AT_byte_size
  | DW_AT_bit_offset
  | DW_AT_bit_size
  | DW_AT_stmt_list
  | DW_AT_low_pc
  | DW_AT_high_pc
  | DW_AT_language
  | DW_AT_discr
  | DW_AT_discr_value
  | DW_AT_visibility
  | DW_AT_import
  | DW_AT_string_length
  | DW_AT_common_reference
  | DW_AT_comp_dir
  | DW_AT_const_value
  | DW_AT_containing_type
  | DW_AT_default_value
  | DW_AT_inline
  | DW_AT_is_optional
  | DW_AT_lower_bound
  | DW_AT_producer
  | DW_AT_prototyped
  | DW_AT_return_addr
  | DW_AT_start_scope
  | DW_AT_bit_stride
  | DW_AT_upper_bound
  | DW_AT_abstract_origin
  | DW_AT_accessibility
  | DW_AT_address_class
  | DW_AT_artificial
  | DW_AT_base_types
  | DW_AT_calling_convention
  | DW_AT_count
  | DW_AT_data_member_location
  | DW_AT_decl_column
  | DW_AT_decl_file
  | DW_AT_decl_line
  | DW_AT_declaration
  | DW_AT_discr_list
  | DW_AT_encoding
  | DW_AT_external
  | DW_AT_frame_base
  | DW_AT_friend
  | DW_AT_identifier_case
  | DW_AT_macro_info
  | DW_AT_namelist_item
  | DW_AT_priority
  | DW_AT_segment
  | DW_AT_specification
  | DW_AT_static_link
  | DW_AT_type
  | DW_AT_use_location
  | DW_AT_variable_parameter
  | DW_AT_virtuality
  | DW_AT_vtable_elem_location
  | DW_AT_allocated
  | DW_AT_associated
  | DW_AT_data_location
  | DW_AT_byte_stride
  | DW_AT_entry_pc
  | DW_AT_use_UTF8
  | DW_AT_extension
  | DW_AT_ranges
  | DW_AT_trampoline
  | DW_AT_call_column
  | DW_AT_call_file
  | DW_AT_call_line
  | DW_AT_description
  | DW_AT_binary_scale
  | DW_AT_decimal_scale
  | DW_AT_small
  | DW_AT_decimal_sign
  | DW_AT_digit_count
  | DW_AT_picture_string
  | DW_AT_mutable
  | DW_AT_threads_scaled
  | DW_AT_explicit
  | DW_AT_object_pointer
  | DW_AT_endianity
  | DW_AT_elemental
  | DW_AT_pure
  | DW_AT_recursive
  | DW_AT_lo_user
  | DW_AT_rust_type_code
  | DW_AT_rust_type_param_index
  | DW_AT_rust_iterator
  | DW_AT_rust_native_type_id
  | DW_AT_hi_user
;;


let dw_at_to_int (a:dw_at) : int =
  match a with
      DW_AT_sibling -> 0x01
    | DW_AT_location -> 0x02
    | DW_AT_name -> 0x03
    | DW_AT_ordering -> 0x09
    | DW_AT_byte_size -> 0x0b
    | DW_AT_bit_offset -> 0x0c
    | DW_AT_bit_size -> 0x0d
    | DW_AT_stmt_list -> 0x10
    | DW_AT_low_pc -> 0x11
    | DW_AT_high_pc -> 0x12
    | DW_AT_language -> 0x13
    | DW_AT_discr -> 0x15
    | DW_AT_discr_value -> 0x16
    | DW_AT_visibility -> 0x17
    | DW_AT_import -> 0x18
    | DW_AT_string_length -> 0x19
    | DW_AT_common_reference -> 0x1a
    | DW_AT_comp_dir -> 0x1b
    | DW_AT_const_value -> 0x1c
    | DW_AT_containing_type -> 0x1d
    | DW_AT_default_value -> 0x1e
    | DW_AT_inline -> 0x20
    | DW_AT_is_optional -> 0x21
    | DW_AT_lower_bound -> 0x22
    | DW_AT_producer -> 0x25
    | DW_AT_prototyped -> 0x27
    | DW_AT_return_addr -> 0x2a
    | DW_AT_start_scope -> 0x2c
    | DW_AT_bit_stride -> 0x2e
    | DW_AT_upper_bound -> 0x2f
    | DW_AT_abstract_origin -> 0x31
    | DW_AT_accessibility -> 0x32
    | DW_AT_address_class -> 0x33
    | DW_AT_artificial -> 0x34
    | DW_AT_base_types -> 0x35
    | DW_AT_calling_convention -> 0x36
    | DW_AT_count -> 0x37
    | DW_AT_data_member_location -> 0x38
    | DW_AT_decl_column -> 0x39
    | DW_AT_decl_file -> 0x3a
    | DW_AT_decl_line -> 0x3b
    | DW_AT_declaration -> 0x3c
    | DW_AT_discr_list -> 0x3d
    | DW_AT_encoding -> 0x3e
    | DW_AT_external -> 0x3f
    | DW_AT_frame_base -> 0x40
    | DW_AT_friend -> 0x41
    | DW_AT_identifier_case -> 0x42
    | DW_AT_macro_info -> 0x43
    | DW_AT_namelist_item -> 0x44
    | DW_AT_priority -> 0x45
    | DW_AT_segment -> 0x46
    | DW_AT_specification -> 0x47
    | DW_AT_static_link -> 0x48
    | DW_AT_type -> 0x49
    | DW_AT_use_location -> 0x4a
    | DW_AT_variable_parameter -> 0x4b
    | DW_AT_virtuality -> 0x4c
    | DW_AT_vtable_elem_location -> 0x4d
    | DW_AT_allocated -> 0x4e
    | DW_AT_associated -> 0x4f
    | DW_AT_data_location -> 0x50
    | DW_AT_byte_stride -> 0x51
    | DW_AT_entry_pc -> 0x52
    | DW_AT_use_UTF8 -> 0x53
    | DW_AT_extension -> 0x54
    | DW_AT_ranges -> 0x55
    | DW_AT_trampoline -> 0x56
    | DW_AT_call_column -> 0x57
    | DW_AT_call_file -> 0x58
    | DW_AT_call_line -> 0x59
    | DW_AT_description -> 0x5a
    | DW_AT_binary_scale -> 0x5b
    | DW_AT_decimal_scale -> 0x5c
    | DW_AT_small -> 0x5d
    | DW_AT_decimal_sign -> 0x5e
    | DW_AT_digit_count -> 0x5f
    | DW_AT_picture_string -> 0x60
    | DW_AT_mutable -> 0x61
    | DW_AT_threads_scaled -> 0x62
    | DW_AT_explicit -> 0x63
    | DW_AT_object_pointer -> 0x64
    | DW_AT_endianity -> 0x65
    | DW_AT_elemental -> 0x66
    | DW_AT_pure -> 0x67
    | DW_AT_recursive -> 0x68
    | DW_AT_lo_user -> 0x2000
    | DW_AT_rust_type_code -> 0x2300
    | DW_AT_rust_type_param_index -> 0x2301
    | DW_AT_rust_iterator -> 0x2302
    | DW_AT_rust_native_type_id -> 0x2303
    | DW_AT_hi_user -> 0x3fff
;;

let dw_at_of_int (i:int) : dw_at =
  match i with
      0x01 -> DW_AT_sibling
    | 0x02 -> DW_AT_location
    | 0x03 -> DW_AT_name
    | 0x09 -> DW_AT_ordering
    | 0x0b -> DW_AT_byte_size
    | 0x0c -> DW_AT_bit_offset
    | 0x0d -> DW_AT_bit_size
    | 0x10 -> DW_AT_stmt_list
    | 0x11 -> DW_AT_low_pc
    | 0x12 -> DW_AT_high_pc
    | 0x13 -> DW_AT_language
    | 0x15 -> DW_AT_discr
    | 0x16 -> DW_AT_discr_value
    | 0x17 -> DW_AT_visibility
    | 0x18 -> DW_AT_import
    | 0x19 -> DW_AT_string_length
    | 0x1a -> DW_AT_common_reference
    | 0x1b -> DW_AT_comp_dir
    | 0x1c -> DW_AT_const_value
    | 0x1d -> DW_AT_containing_type
    | 0x1e -> DW_AT_default_value
    | 0x20 -> DW_AT_inline
    | 0x21 -> DW_AT_is_optional
    | 0x22 -> DW_AT_lower_bound
    | 0x25 -> DW_AT_producer
    | 0x27 -> DW_AT_prototyped
    | 0x2a -> DW_AT_return_addr
    | 0x2c -> DW_AT_start_scope
    | 0x2e -> DW_AT_bit_stride
    | 0x2f -> DW_AT_upper_bound
    | 0x31 -> DW_AT_abstract_origin
    | 0x32 -> DW_AT_accessibility
    | 0x33 -> DW_AT_address_class
    | 0x34 -> DW_AT_artificial
    | 0x35 -> DW_AT_base_types
    | 0x36 -> DW_AT_calling_convention
    | 0x37 -> DW_AT_count
    | 0x38 -> DW_AT_data_member_location
    | 0x39 -> DW_AT_decl_column
    | 0x3a -> DW_AT_decl_file
    | 0x3b -> DW_AT_decl_line
    | 0x3c -> DW_AT_declaration
    | 0x3d -> DW_AT_discr_list
    | 0x3e -> DW_AT_encoding
    | 0x3f -> DW_AT_external
    | 0x40 -> DW_AT_frame_base
    | 0x41 -> DW_AT_friend
    | 0x42 -> DW_AT_identifier_case
    | 0x43 -> DW_AT_macro_info
    | 0x44 -> DW_AT_namelist_item
    | 0x45 -> DW_AT_priority
    | 0x46 -> DW_AT_segment
    | 0x47 -> DW_AT_specification
    | 0x48 -> DW_AT_static_link
    | 0x49 -> DW_AT_type
    | 0x4a -> DW_AT_use_location
    | 0x4b -> DW_AT_variable_parameter
    | 0x4c -> DW_AT_virtuality
    | 0x4d -> DW_AT_vtable_elem_location
    | 0x4e -> DW_AT_allocated
    | 0x4f -> DW_AT_associated
    | 0x50 -> DW_AT_data_location
    | 0x51 -> DW_AT_byte_stride
    | 0x52 -> DW_AT_entry_pc
    | 0x53 -> DW_AT_use_UTF8
    | 0x54 -> DW_AT_extension
    | 0x55 -> DW_AT_ranges
    | 0x56 -> DW_AT_trampoline
    | 0x57 -> DW_AT_call_column
    | 0x58 -> DW_AT_call_file
    | 0x59 -> DW_AT_call_line
    | 0x5a -> DW_AT_description
    | 0x5b -> DW_AT_binary_scale
    | 0x5c -> DW_AT_decimal_scale
    | 0x5d -> DW_AT_small
    | 0x5e -> DW_AT_decimal_sign
    | 0x5f -> DW_AT_digit_count
    | 0x60 -> DW_AT_picture_string
    | 0x61 -> DW_AT_mutable
    | 0x62 -> DW_AT_threads_scaled
    | 0x63 -> DW_AT_explicit
    | 0x64 -> DW_AT_object_pointer
    | 0x65 -> DW_AT_endianity
    | 0x66 -> DW_AT_elemental
    | 0x67 -> DW_AT_pure
    | 0x68 -> DW_AT_recursive
    | 0x2000 -> DW_AT_lo_user
    | 0x2300 -> DW_AT_rust_type_code
    | 0x2301 -> DW_AT_rust_type_param_index
    | 0x2302 -> DW_AT_rust_iterator
    | 0x2303 -> DW_AT_rust_native_type_id
    | 0x3fff -> DW_AT_hi_user
    | _ -> bug () "bad DWARF attribute code: 0x%x" i
;;

let dw_at_to_string (a:dw_at) : string =
  match a with
      DW_AT_sibling -> "DW_AT_sibling"
    | DW_AT_location -> "DW_AT_location"
    | DW_AT_name -> "DW_AT_name"
    | DW_AT_ordering -> "DW_AT_ordering"
    | DW_AT_byte_size -> "DW_AT_byte_size"
    | DW_AT_bit_offset -> "DW_AT_bit_offset"
    | DW_AT_bit_size -> "DW_AT_bit_size"
    | DW_AT_stmt_list -> "DW_AT_stmt_list"
    | DW_AT_low_pc -> "DW_AT_low_pc"
    | DW_AT_high_pc -> "DW_AT_high_pc"
    | DW_AT_language -> "DW_AT_language"
    | DW_AT_discr -> "DW_AT_discr"
    | DW_AT_discr_value -> "DW_AT_discr_value"
    | DW_AT_visibility -> "DW_AT_visibility"
    | DW_AT_import -> "DW_AT_import"
    | DW_AT_string_length -> "DW_AT_string_length"
    | DW_AT_common_reference -> "DW_AT_common_reference"
    | DW_AT_comp_dir -> "DW_AT_comp_dir"
    | DW_AT_const_value -> "DW_AT_const_value"
    | DW_AT_containing_type -> "DW_AT_containing_type"
    | DW_AT_default_value -> "DW_AT_default_value"
    | DW_AT_inline -> "DW_AT_inline"
    | DW_AT_is_optional -> "DW_AT_is_optional"
    | DW_AT_lower_bound -> "DW_AT_lower_bound"
    | DW_AT_producer -> "DW_AT_producer"
    | DW_AT_prototyped -> "DW_AT_prototyped"
    | DW_AT_return_addr -> "DW_AT_return_addr"
    | DW_AT_start_scope -> "DW_AT_start_scope"
    | DW_AT_bit_stride -> "DW_AT_bit_stride"
    | DW_AT_upper_bound -> "DW_AT_upper_bound"
    | DW_AT_abstract_origin -> "DW_AT_abstract_origin"
    | DW_AT_accessibility -> "DW_AT_accessibility"
    | DW_AT_address_class -> "DW_AT_address_class"
    | DW_AT_artificial -> "DW_AT_artificial"
    | DW_AT_base_types -> "DW_AT_base_types"
    | DW_AT_calling_convention -> "DW_AT_calling_convention"
    | DW_AT_count -> "DW_AT_count"
    | DW_AT_data_member_location -> "DW_AT_data_member_location"
    | DW_AT_decl_column -> "DW_AT_decl_column"
    | DW_AT_decl_file -> "DW_AT_decl_file"
    | DW_AT_decl_line -> "DW_AT_decl_line"
    | DW_AT_declaration -> "DW_AT_declaration"
    | DW_AT_discr_list -> "DW_AT_discr_list"
    | DW_AT_encoding -> "DW_AT_encoding"
    | DW_AT_external -> "DW_AT_external"
    | DW_AT_frame_base -> "DW_AT_frame_base"
    | DW_AT_friend -> "DW_AT_friend"
    | DW_AT_identifier_case -> "DW_AT_identifier_case"
    | DW_AT_macro_info -> "DW_AT_macro_info"
    | DW_AT_namelist_item -> "DW_AT_namelist_item"
    | DW_AT_priority -> "DW_AT_priority"
    | DW_AT_segment -> "DW_AT_segment"
    | DW_AT_specification -> "DW_AT_specification"
    | DW_AT_static_link -> "DW_AT_static_link"
    | DW_AT_type -> "DW_AT_type"
    | DW_AT_use_location -> "DW_AT_use_location"
    | DW_AT_variable_parameter -> "DW_AT_variable_parameter"
    | DW_AT_virtuality -> "DW_AT_virtuality"
    | DW_AT_vtable_elem_location -> "DW_AT_vtable_elem_location"
    | DW_AT_allocated -> "DW_AT_allocated"
    | DW_AT_associated -> "DW_AT_associated"
    | DW_AT_data_location -> "DW_AT_data_location"
    | DW_AT_byte_stride -> "DW_AT_byte_stride"
    | DW_AT_entry_pc -> "DW_AT_entry_pc"
    | DW_AT_use_UTF8 -> "DW_AT_use_UTF8"
    | DW_AT_extension -> "DW_AT_extension"
    | DW_AT_ranges -> "DW_AT_ranges"
    | DW_AT_trampoline -> "DW_AT_trampoline"
    | DW_AT_call_column -> "DW_AT_call_column"
    | DW_AT_call_file -> "DW_AT_call_file"
    | DW_AT_call_line -> "DW_AT_call_line"
    | DW_AT_description -> "DW_AT_description"
    | DW_AT_binary_scale -> "DW_AT_binary_scale"
    | DW_AT_decimal_scale -> "DW_AT_decimal_scale"
    | DW_AT_small -> "DW_AT_small"
    | DW_AT_decimal_sign -> "DW_AT_decimal_sign"
    | DW_AT_digit_count -> "DW_AT_digit_count"
    | DW_AT_picture_string -> "DW_AT_picture_string"
    | DW_AT_mutable -> "DW_AT_mutable"
    | DW_AT_threads_scaled -> "DW_AT_threads_scaled"
    | DW_AT_explicit -> "DW_AT_explicit"
    | DW_AT_object_pointer -> "DW_AT_object_pointer"
    | DW_AT_endianity -> "DW_AT_endianity"
    | DW_AT_elemental -> "DW_AT_elemental"
    | DW_AT_pure -> "DW_AT_pure"
    | DW_AT_recursive -> "DW_AT_recursive"
    | DW_AT_lo_user -> "DW_AT_lo_user"
    | DW_AT_rust_type_code -> "DW_AT_rust_type_code"
    | DW_AT_rust_type_param_index -> "DW_AT_rust_type_param_index"
    | DW_AT_rust_iterator -> "DW_AT_rust_iterator"
    | DW_AT_rust_native_type_id -> "DW_AT_native_type_id"
    | DW_AT_hi_user -> "DW_AT_hi_user"
;;

(*
 * We encode our 'built-in types' using DW_TAG_pointer_type and various
 * DW_AT_pointer_type_codes. This seems to be more gdb-compatible than
 * the DWARF-recommended way of using DW_TAG_unspecified_type.
 *)
type dw_rust_type =
    DW_RUST_type_param
  | DW_RUST_nil
  | DW_RUST_vec
  | DW_RUST_chan
  | DW_RUST_port
  | DW_RUST_task
  | DW_RUST_type
  | DW_RUST_native
;;

let dw_rust_type_to_int (pt:dw_rust_type) : int =
  match pt with
      DW_RUST_type_param -> 0x1
    | DW_RUST_nil -> 0x2
    | DW_RUST_vec -> 0x3
    | DW_RUST_chan -> 0x4
    | DW_RUST_port -> 0x5
    | DW_RUST_task -> 0x6
    | DW_RUST_type -> 0x7
    | DW_RUST_native -> 0x8
;;

let dw_rust_type_of_int (i:int) : dw_rust_type =
  match i with
      0x1 -> DW_RUST_type_param
    | 0x2 -> DW_RUST_nil
    | 0x3 -> DW_RUST_vec
    | 0x4 -> DW_RUST_chan
    | 0x5 -> DW_RUST_port
    | 0x6 -> DW_RUST_task
    | 0x7 -> DW_RUST_type
    | 0x8 -> DW_RUST_native
    | _ -> bug () "bad DWARF rust-pointer-type code: %d" i
;;

type dw_ate =
      DW_ATE_address
    | DW_ATE_boolean
    | DW_ATE_complex_float
    | DW_ATE_float
    | DW_ATE_signed
    | DW_ATE_signed_char
    | DW_ATE_unsigned
    | DW_ATE_unsigned_char
    | DW_ATE_imaginary_float
    | DW_ATE_packed_decimal
    | DW_ATE_numeric_string
    | DW_ATE_edited
    | DW_ATE_signed_fixed
    | DW_ATE_unsigned_fixed
    | DW_ATE_decimal_float
    | DW_ATE_lo_user
    | DW_ATE_hi_user
;;

let dw_ate_to_int (ate:dw_ate) : int =
  match ate with
      DW_ATE_address -> 0x01
    | DW_ATE_boolean -> 0x02
    | DW_ATE_complex_float -> 0x03
    | DW_ATE_float -> 0x04
    | DW_ATE_signed -> 0x05
    | DW_ATE_signed_char -> 0x06
    | DW_ATE_unsigned -> 0x07
    | DW_ATE_unsigned_char -> 0x08
    | DW_ATE_imaginary_float -> 0x09
    | DW_ATE_packed_decimal -> 0x0a
    | DW_ATE_numeric_string -> 0x0b
    | DW_ATE_edited -> 0x0c
    | DW_ATE_signed_fixed -> 0x0d
    | DW_ATE_unsigned_fixed -> 0x0e
    | DW_ATE_decimal_float -> 0x0f
    | DW_ATE_lo_user -> 0x80
    | DW_ATE_hi_user -> 0xff
;;

let dw_ate_of_int (i:int) : dw_ate =
  match i with
      0x01 -> DW_ATE_address
    | 0x02 -> DW_ATE_boolean
    | 0x03 -> DW_ATE_complex_float
    | 0x04 -> DW_ATE_float
    | 0x05 -> DW_ATE_signed
    | 0x06 -> DW_ATE_signed_char
    | 0x07 -> DW_ATE_unsigned
    | 0x08 -> DW_ATE_unsigned_char
    | 0x09 -> DW_ATE_imaginary_float
    | 0x0a -> DW_ATE_packed_decimal
    | 0x0b -> DW_ATE_numeric_string
    | 0x0c -> DW_ATE_edited
    | 0x0d -> DW_ATE_signed_fixed
    | 0x0e -> DW_ATE_unsigned_fixed
    | 0x0f -> DW_ATE_decimal_float
    | 0x80 -> DW_ATE_lo_user
    | 0xff -> DW_ATE_hi_user
    | _ -> bug () "bad DWARF attribute-encoding code: %d" i
;;

type dw_form =
  | DW_FORM_addr
  | DW_FORM_block2
  | DW_FORM_block4
  | DW_FORM_data2
  | DW_FORM_data4
  | DW_FORM_data8
  | DW_FORM_string
  | DW_FORM_block
  | DW_FORM_block1
  | DW_FORM_data1
  | DW_FORM_flag
  | DW_FORM_sdata
  | DW_FORM_strp
  | DW_FORM_udata
  | DW_FORM_ref_addr
  | DW_FORM_ref1
  | DW_FORM_ref2
  | DW_FORM_ref4
  | DW_FORM_ref8
  | DW_FORM_ref_udata
  | DW_FORM_indirect
;;


let dw_form_to_int (f:dw_form) : int =
  match f with
    | DW_FORM_addr -> 0x01
    | DW_FORM_block2 -> 0x03
    | DW_FORM_block4 -> 0x04
    | DW_FORM_data2 -> 0x05
    | DW_FORM_data4 -> 0x06
    | DW_FORM_data8 -> 0x07
    | DW_FORM_string -> 0x08
    | DW_FORM_block -> 0x09
    | DW_FORM_block1 -> 0x0a
    | DW_FORM_data1 -> 0x0b
    | DW_FORM_flag -> 0x0c
    | DW_FORM_sdata -> 0x0d
    | DW_FORM_strp -> 0x0e
    | DW_FORM_udata -> 0x0f
    | DW_FORM_ref_addr -> 0x10
    | DW_FORM_ref1 -> 0x11
    | DW_FORM_ref2 -> 0x12
    | DW_FORM_ref4 -> 0x13
    | DW_FORM_ref8 -> 0x14
    | DW_FORM_ref_udata -> 0x15
    | DW_FORM_indirect -> 0x16
;;

let dw_form_of_int (i:int) : dw_form =
  match i with
    | 0x01 -> DW_FORM_addr
    | 0x03 -> DW_FORM_block2
    | 0x04 -> DW_FORM_block4
    | 0x05 -> DW_FORM_data2
    | 0x06 -> DW_FORM_data4
    | 0x07 -> DW_FORM_data8
    | 0x08 -> DW_FORM_string
    | 0x09 -> DW_FORM_block
    | 0x0a -> DW_FORM_block1
    | 0x0b -> DW_FORM_data1
    | 0x0c -> DW_FORM_flag
    | 0x0d -> DW_FORM_sdata
    | 0x0e -> DW_FORM_strp
    | 0x0f -> DW_FORM_udata
    | 0x10 -> DW_FORM_ref_addr
    | 0x11 -> DW_FORM_ref1
    | 0x12 -> DW_FORM_ref2
    | 0x13 -> DW_FORM_ref4
    | 0x14 -> DW_FORM_ref8
    | 0x15 -> DW_FORM_ref_udata
    | 0x16 -> DW_FORM_indirect
    | _ -> bug () "bad DWARF form code: 0x%x" i
;;

let dw_form_to_string (f:dw_form) : string =
  match f with
    | DW_FORM_addr -> "DW_FORM_addr"
    | DW_FORM_block2 -> "DW_FORM_block2"
    | DW_FORM_block4 -> "DW_FORM_block4"
    | DW_FORM_data2 -> "DW_FORM_data2"
    | DW_FORM_data4 -> "DW_FORM_data4"
    | DW_FORM_data8 -> "DW_FORM_data8"
    | DW_FORM_string -> "DW_FORM_string"
    | DW_FORM_block -> "DW_FORM_block"
    | DW_FORM_block1 -> "DW_FORM_block1"
    | DW_FORM_data1 -> "DW_FORM_data1"
    | DW_FORM_flag -> "DW_FORM_flag"
    | DW_FORM_sdata -> "DW_FORM_sdata"
    | DW_FORM_strp -> "DW_FORM_strp"
    | DW_FORM_udata -> "DW_FORM_udata"
    | DW_FORM_ref_addr -> "DW_FORM_ref_addr"
    | DW_FORM_ref1 -> "DW_FORM_ref1"
    | DW_FORM_ref2 -> "DW_FORM_ref2"
    | DW_FORM_ref4 -> "DW_FORM_ref4"
    | DW_FORM_ref8 -> "DW_FORM_ref8"
    | DW_FORM_ref_udata -> "DW_FORM_ref_udata"
    | DW_FORM_indirect -> "DW_FORM_indirect"
;;

type dw_op =
    DW_OP_lit of int
  | DW_OP_addr of Asm.expr64
  | DW_OP_const1u of Asm.expr64
  | DW_OP_const1s of Asm.expr64
  | DW_OP_const2u of Asm.expr64
  | DW_OP_const2s of Asm.expr64
  | DW_OP_const4u of Asm.expr64
  | DW_OP_const4s of Asm.expr64
  | DW_OP_const8u of Asm.expr64
  | DW_OP_const8s of Asm.expr64
  | DW_OP_constu of Asm.expr64
  | DW_OP_consts of Asm.expr64
  | DW_OP_fbreg of Asm.expr64
  | DW_OP_reg of int
  | DW_OP_regx of Asm.expr64
  | DW_OP_breg of (int * Asm.expr64)
  | DW_OP_bregx of (Asm.expr64 * Asm.expr64)
  | DW_OP_dup
  | DW_OP_drop
  | DW_OP_pick of Asm.expr64
  | DW_OP_over
  | DW_OP_swap
  | DW_OP_rot
  | DW_OP_piece of Asm.expr64
  | DW_OP_bit_piece of (Asm.expr64 * Asm.expr64)
  | DW_OP_deref
  | DW_OP_deref_size of Asm.expr64
  | DW_OP_xderef
  | DW_OP_xderef_size of Asm.expr64
  | DW_OP_push_object_address
  | DW_OP_form_tls_address
  | DW_OP_call_frame_cfa
  | DW_OP_abs
  | DW_OP_and
  | DW_OP_div
  | DW_OP_minus
  | DW_OP_mod
  | DW_OP_mul
  | DW_OP_neg
  | DW_OP_not
  | DW_OP_or
  | DW_OP_plus
  | DW_OP_plus_uconst of Asm.expr64
  | DW_OP_shl
  | DW_OP_shr
  | DW_OP_shra
  | DW_OP_xor
  | DW_OP_le
  | DW_OP_ge
  | DW_OP_eq
  | DW_OP_lt
  | DW_OP_gt
  | DW_OP_ne
  | DW_OP_skip of Asm.expr64
  | DW_OP_bra of Asm.expr64
  | DW_OP_call2 of Asm.expr64
  | DW_OP_call4 of Asm.expr64
  | DW_OP_call_ref of Asm.expr64
  | DW_OP_nop
;;

let dw_op_to_frag (abi:Abi.abi) (op:dw_op) : Asm.frag =
  match op with

      DW_OP_addr e -> SEQ [| BYTE 0x03; WORD (abi.Abi.abi_word_ty, e) |]
    | DW_OP_deref -> BYTE 0x06
    | DW_OP_const1u e -> SEQ [| BYTE 0x08; WORD (TY_u8, e) |]
    | DW_OP_const1s e -> SEQ [| BYTE 0x09; WORD (TY_i8, e) |]
    | DW_OP_const2u e -> SEQ [| BYTE 0x0a; WORD (TY_u16, e) |]
    | DW_OP_const2s e -> SEQ [| BYTE 0x0b; WORD (TY_i16, e) |]
    | DW_OP_const4u e -> SEQ [| BYTE 0x0c; WORD (TY_u32, e) |]
    | DW_OP_const4s e -> SEQ [| BYTE 0x0d; WORD (TY_i32, e) |]
    | DW_OP_const8u e -> SEQ [| BYTE 0x0e; WORD (TY_u64, e) |]
    | DW_OP_const8s e -> SEQ [| BYTE 0x0f; WORD (TY_i64, e) |]
    | DW_OP_constu e -> SEQ [| BYTE 0x10; ULEB128 e |]
    | DW_OP_consts e -> SEQ [| BYTE 0x11; SLEB128 e |]
    | DW_OP_dup -> BYTE 0x12
    | DW_OP_drop -> BYTE 0x13
    | DW_OP_over -> BYTE 0x14
    | DW_OP_pick e -> SEQ [| BYTE 0x15; WORD (TY_u8, e) |]
    | DW_OP_swap -> BYTE 0x16
    | DW_OP_rot -> BYTE 0x17
    | DW_OP_xderef -> BYTE 0x18
    | DW_OP_abs -> BYTE 0x19
    | DW_OP_and -> BYTE 0x1a
    | DW_OP_div -> BYTE 0x1b
    | DW_OP_minus -> BYTE 0x1c
    | DW_OP_mod -> BYTE 0x1d
    | DW_OP_mul -> BYTE 0x1e
    | DW_OP_neg -> BYTE 0x1f
    | DW_OP_not -> BYTE 0x20
    | DW_OP_or -> BYTE 0x21
    | DW_OP_plus -> BYTE 0x22
    | DW_OP_plus_uconst e -> SEQ [| BYTE 0x23; ULEB128 e |]
    | DW_OP_shl -> BYTE 0x24
    | DW_OP_shr -> BYTE 0x25
    | DW_OP_shra -> BYTE 0x26
    | DW_OP_xor -> BYTE 0x27
    | DW_OP_skip e -> SEQ [| BYTE 0x2f; WORD (TY_i16, e) |]
    | DW_OP_bra e -> SEQ [| BYTE 0x28; WORD (TY_i16, e) |]
    | DW_OP_eq -> BYTE 0x29
    | DW_OP_ge -> BYTE 0x2a
    | DW_OP_gt -> BYTE 0x2b
    | DW_OP_le -> BYTE 0x2c
    | DW_OP_lt -> BYTE 0x2d
    | DW_OP_ne -> BYTE 0x2e

    | DW_OP_lit i ->
        assert (0 <= i && i < 32);
        BYTE (i + 0x30)

    | DW_OP_reg i ->
        assert (0 <= i && i < 32);
        BYTE (i + 0x50)

    | DW_OP_breg (i, e) ->
        assert (0 <= i && i < 32);
        SEQ [| BYTE (i + 0x70); SLEB128 e |]

    | DW_OP_regx e -> SEQ [| BYTE 0x90; ULEB128 e|]
    | DW_OP_fbreg e -> SEQ [| BYTE 0x91; SLEB128 e |]
    | DW_OP_bregx (r, off) -> SEQ [| BYTE 0x92; ULEB128 r; SLEB128 off |]
    | DW_OP_piece e -> SEQ [| BYTE 0x93; ULEB128 e |]
    | DW_OP_deref_size e -> SEQ [| BYTE 0x94; WORD (TY_u8, e) |]
    | DW_OP_xderef_size e -> SEQ [| BYTE 0x95; WORD (TY_u8, e) |]
    | DW_OP_nop -> BYTE 0x96
    | DW_OP_push_object_address -> BYTE 0x97
    | DW_OP_call2 e -> SEQ [| BYTE 0x98; WORD (TY_u16, e) |]
    | DW_OP_call4 e -> SEQ [| BYTE 0x99; WORD (TY_u32, e) |]
    | DW_OP_call_ref e -> SEQ [| BYTE 0x9a; WORD (abi.Abi.abi_word_ty, e) |]
    | DW_OP_form_tls_address -> BYTE 0x9b
    | DW_OP_call_frame_cfa -> BYTE 0x9c
    | DW_OP_bit_piece (sz, off) ->
        SEQ [| BYTE 0x9d; ULEB128 sz; ULEB128 off |]
;;

type dw_lns =
      DW_LNS_copy
    | DW_LNS_advance_pc
    | DW_LNS_advance_line
    | DW_LNS_set_file
    | DW_LNS_set_column
    | DW_LNS_negage_stmt
    | DW_LNS_set_basic_block
    | DW_LNS_const_add_pc
    | DW_LNS_fixed_advance_pc
    | DW_LNS_set_prologue_end
    | DW_LNS_set_epilogue_begin
    | DW_LNS_set_isa
;;

let int_to_dw_lns i =
  match i with
      1 -> DW_LNS_copy
    | 2 -> DW_LNS_advance_pc
    | 3 -> DW_LNS_advance_line
    | 4 -> DW_LNS_set_file
    | 5 -> DW_LNS_set_column
    | 6 -> DW_LNS_negage_stmt
    | 7 -> DW_LNS_set_basic_block
    | 8 -> DW_LNS_const_add_pc
    | 9 -> DW_LNS_fixed_advance_pc
    | 10 -> DW_LNS_set_prologue_end
    | 11 -> DW_LNS_set_epilogue_begin
    | 12 -> DW_LNS_set_isa
    | _ -> bug () "Internal logic error: (Dwarf.int_to_dw_lns %d)" i
;;

let dw_lns_to_int lns =
  match lns with
      DW_LNS_copy -> 1
    | DW_LNS_advance_pc -> 2
    | DW_LNS_advance_line -> 3
    | DW_LNS_set_file -> 4
    | DW_LNS_set_column -> 5
    | DW_LNS_negage_stmt -> 6
    | DW_LNS_set_basic_block -> 7
    | DW_LNS_const_add_pc -> 8
    | DW_LNS_fixed_advance_pc -> 9
    | DW_LNS_set_prologue_end -> 10
    | DW_LNS_set_epilogue_begin -> 11
    | DW_LNS_set_isa -> 12
;;

let max_dw_lns = 12;;

let dw_lns_arity lns =
  match lns with
      DW_LNS_copy -> 0
    | DW_LNS_advance_pc -> 1
    | DW_LNS_advance_line -> 1
    | DW_LNS_set_file -> 1
    | DW_LNS_set_column -> 1
    | DW_LNS_negage_stmt -> 0
    | DW_LNS_set_basic_block -> 0
    | DW_LNS_const_add_pc -> 0
    | DW_LNS_fixed_advance_pc -> 1
    | DW_LNS_set_prologue_end -> 0
    | DW_LNS_set_epilogue_begin -> 0
    | DW_LNS_set_isa -> 1
;;

type debug_records =
    {
      debug_aranges: Asm.frag;
      debug_pubnames: Asm.frag;
      debug_info: Asm.frag;
      debug_abbrev: Asm.frag;
      debug_line: Asm.frag;
      debug_frame: Asm.frag;
    }

type abbrev = (dw_tag * dw_children * ((dw_at * dw_form) array));;

let (abbrev_crate_cu:abbrev) =
   (DW_TAG_compile_unit, DW_CHILDREN_yes,
    [|
     (DW_AT_producer, DW_FORM_string);
     (DW_AT_language, DW_FORM_data4);
     (DW_AT_name, DW_FORM_string);
     (DW_AT_comp_dir, DW_FORM_string);
     (DW_AT_low_pc, DW_FORM_addr);
     (DW_AT_high_pc, DW_FORM_addr);
     (DW_AT_use_UTF8, DW_FORM_flag)
    |])
 ;;

let (abbrev_meta:abbrev) =
  (DW_TAG_rust_meta, DW_CHILDREN_no,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_const_value, DW_FORM_string)
   |])
;;

let (abbrev_srcfile_cu:abbrev) =
  (DW_TAG_compile_unit, DW_CHILDREN_yes,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_comp_dir, DW_FORM_string);
     (DW_AT_low_pc, DW_FORM_addr);
     (DW_AT_high_pc, DW_FORM_addr);
   |])
;;


let (abbrev_module:abbrev) =
  (DW_TAG_module, DW_CHILDREN_yes,
   [|
     (DW_AT_name, DW_FORM_string);
   |])
;;

let (abbrev_subprogram:abbrev) =
  (DW_TAG_subprogram, DW_CHILDREN_yes,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_type, DW_FORM_ref_addr);
     (DW_AT_low_pc, DW_FORM_addr);
     (DW_AT_high_pc, DW_FORM_addr);
     (DW_AT_frame_base, DW_FORM_block1);
     (DW_AT_return_addr, DW_FORM_block1);
     (DW_AT_mutable, DW_FORM_flag);
     (DW_AT_pure, DW_FORM_flag);
   |])
;;

let (abbrev_typedef:abbrev) =
  (DW_TAG_typedef, DW_CHILDREN_yes,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_type, DW_FORM_ref_addr)
   |])
;;

let (abbrev_lexical_block:abbrev) =
  (DW_TAG_lexical_block, DW_CHILDREN_yes,
   [|
     (DW_AT_low_pc, DW_FORM_addr);
     (DW_AT_high_pc, DW_FORM_addr);
   |])
;;

let (abbrev_variable:abbrev) =
  (DW_TAG_variable, DW_CHILDREN_no,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_location, DW_FORM_block1);
     (DW_AT_type, DW_FORM_ref_addr)
   |])
;;

(* NB: must have same abbrev-body as abbrev_variable. *)
let (abbrev_formal:abbrev) =
  (DW_TAG_formal_parameter, DW_CHILDREN_no,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_location, DW_FORM_block1);
     (DW_AT_type, DW_FORM_ref_addr)
   |])
;;

let (abbrev_unspecified_anon_structure_type:abbrev) =
  (DW_TAG_structure_type, DW_CHILDREN_no,
   [|
     (DW_AT_declaration, DW_FORM_flag);
   |])
;;

let (abbrev_unspecified_structure_type:abbrev) =
  (DW_TAG_structure_type, DW_CHILDREN_no,
   [|
     (DW_AT_rust_type_code, DW_FORM_data1);
     (DW_AT_declaration, DW_FORM_flag);
   |])
;;

let (abbrev_unspecified_pointer_type:abbrev) =
  (DW_TAG_pointer_type, DW_CHILDREN_no,
   [|
     (DW_AT_rust_type_code, DW_FORM_data1);
     (DW_AT_declaration, DW_FORM_flag);
     (DW_AT_type, DW_FORM_ref_addr)
   |])
;;

let (abbrev_native_pointer_type:abbrev) =
  (DW_TAG_pointer_type, DW_CHILDREN_no,
   [|
     (DW_AT_rust_type_code, DW_FORM_data1);
     (DW_AT_rust_native_type_id, DW_FORM_data4)
   |])
;;

let (abbrev_rust_type_param:abbrev) =
  (DW_TAG_pointer_type, DW_CHILDREN_no,
   [|
     (DW_AT_rust_type_code, DW_FORM_data1);
     (DW_AT_rust_type_param_index, DW_FORM_data4);
     (DW_AT_mutable, DW_FORM_flag);
     (DW_AT_pure, DW_FORM_flag);
   |])
;;

let (abbrev_rust_type_param_decl:abbrev) =
  (DW_TAG_formal_parameter, DW_CHILDREN_no,
   [|
     (DW_AT_rust_type_code, DW_FORM_data1);
     (DW_AT_name, DW_FORM_string);
     (DW_AT_rust_type_param_index, DW_FORM_data4);
     (DW_AT_mutable, DW_FORM_flag);
     (DW_AT_pure, DW_FORM_flag);
   |])
;;

let (abbrev_base_type:abbrev) =
  (DW_TAG_base_type, DW_CHILDREN_no,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_encoding, DW_FORM_data1);
     (DW_AT_byte_size, DW_FORM_data1)
   |])
;;

let (abbrev_alias_slot:abbrev) =
  (DW_TAG_reference_type, DW_CHILDREN_no,
   [|
     (DW_AT_type, DW_FORM_ref_addr);
   |])
;;

(* FIXME: Perverse, but given dwarf's vocabulary it seems at least plausible
 * that a "mutable const type" is a correct way of saying "mutable". 
 * Or else we make up our own. Revisit perhaps.
 *)

let (abbrev_mutable_type:abbrev) =
  (DW_TAG_const_type, DW_CHILDREN_no,
   [|
     (DW_AT_type, DW_FORM_ref_addr);
     (DW_AT_mutable, DW_FORM_flag);
   |])
;;

let (abbrev_exterior_type:abbrev) =
  (DW_TAG_pointer_type, DW_CHILDREN_no,
   [|
     (DW_AT_type, DW_FORM_ref_addr);
     (DW_AT_data_location, DW_FORM_block1);
   |])
;;

let (abbrev_struct_type:abbrev) =
  (DW_TAG_structure_type, DW_CHILDREN_yes,
   [|
     (DW_AT_byte_size, DW_FORM_block4)
   |])
;;

let (abbrev_struct_type_member:abbrev) =
  (DW_TAG_member, DW_CHILDREN_no,
   [|
     (DW_AT_name, DW_FORM_string);
     (DW_AT_type, DW_FORM_ref_addr);
     (DW_AT_data_member_location, DW_FORM_block4);
     (DW_AT_byte_size, DW_FORM_block4)
   |])
;;

let (abbrev_variant_part:abbrev) =
  (DW_TAG_variant_part, DW_CHILDREN_yes,
   [|
     (DW_AT_discr, DW_FORM_ref_addr)
   |])
;;


let (abbrev_variant:abbrev) =
  (DW_TAG_variant, DW_CHILDREN_yes,
   [|
     (DW_AT_discr_value, DW_FORM_udata)
   |])
;;

let (abbrev_subroutine_type:abbrev) =
  (DW_TAG_subroutine_type, DW_CHILDREN_yes,
   [|
     (DW_AT_type, DW_FORM_ref_addr); (* NB: output type. *)
       (DW_AT_mutable, DW_FORM_flag);
       (DW_AT_pure, DW_FORM_flag);
       (DW_AT_rust_iterator, DW_FORM_flag);
     |])
;;

let (abbrev_formal_type:abbrev) =
  (DW_TAG_formal_parameter, DW_CHILDREN_no,
   [|
     (DW_AT_type, DW_FORM_ref_addr)
   |])
;;


let (abbrev_obj_subroutine_type:abbrev) =
    (DW_TAG_subroutine_type, DW_CHILDREN_yes,
     [|
       (DW_AT_name, DW_FORM_string);
       (DW_AT_type, DW_FORM_ref_addr); (* NB: output type. *)
       (DW_AT_mutable, DW_FORM_flag);
       (DW_AT_pure, DW_FORM_flag);
       (DW_AT_rust_iterator, DW_FORM_flag);
     |])
;;

let (abbrev_obj_type:abbrev) =
    (DW_TAG_interface_type, DW_CHILDREN_yes,
     [|
       (DW_AT_mutable, DW_FORM_flag);
       (DW_AT_pure, DW_FORM_flag);
     |])
;;

let (abbrev_string_type:abbrev) =
    (DW_TAG_string_type, DW_CHILDREN_no,
     [|
       (DW_AT_string_length, DW_FORM_block1);
       (DW_AT_data_location, DW_FORM_block1);
     |])
;;


let prepend lref x = lref := x :: (!lref)
;;


let dwarf_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    (path:Ast.name_component Stack.t)
    (cu_info_section_fixup:fixup)
    (cu_aranges:(frag list) ref)
    (cu_pubnames:(frag list) ref)
    (cu_infos:(frag list) ref)
    (cu_abbrevs:(frag list) ref)
    (cu_lines:(frag list) ref)
    (cu_frames:(frag list) ref)
    : Walk.visitor =

  let (abi:Abi.abi) = cx.ctxt_abi in
  let (word_sz:int64) = abi.Abi.abi_word_sz in
  let (word_sz_int:int) = Int64.to_int word_sz in
  let (word_bits:Il.bits) = abi.Abi.abi_word_bits in
  let (word_ty_mach:ty_mach) =
    match word_bits with
        Il.Bits8 -> TY_u8
      | Il.Bits16 -> TY_u16
      | Il.Bits32 -> TY_u32
      | Il.Bits64 -> TY_u64
  in
  let (signed_word_ty_mach:ty_mach) =
    match word_bits with
        Il.Bits8 -> TY_i8
      | Il.Bits16 -> TY_i16
      | Il.Bits32 -> TY_i32
      | Il.Bits64 -> TY_i64
  in

  let iso_stack = Stack.create () in

  let path_name _ = Fmt.fmt_to_str Ast.fmt_name (Walk.path_to_name path) in

  let (abbrev_table:(abbrev, int) Hashtbl.t) = Hashtbl.create 0 in

  let uleb i = ULEB128 (IMM (Int64.of_int i)) in

  let get_abbrev_code
      (ab:abbrev)
      : int =
    if Hashtbl.mem abbrev_table ab
    then Hashtbl.find abbrev_table ab
    else
      let n = (Hashtbl.length abbrev_table) + 1 in
      let (tag, children, attrs) = ab in
      let attr_ulebs = Array.create ((Array.length attrs) * 2) MARK in
        for i = 0 to (Array.length attrs) - 1 do
          let (attr, form) = attrs.(i) in
            attr_ulebs.(2*i) <- uleb (dw_at_to_int attr);
            attr_ulebs.((2*i)+1) <- uleb (dw_form_to_int form)
        done;
        let ab_frag =
          (SEQ [|
             uleb n;
             uleb (dw_tag_to_int tag);
             BYTE (dw_children_to_int children);
             SEQ attr_ulebs;
             uleb 0; uleb 0;
           |])
        in
          prepend cu_abbrevs ab_frag;
          htab_put abbrev_table ab n;
          n
  in

  let (curr_cu_aranges:(frag list) ref) = ref [] in
  let (curr_cu_pubnames:(frag list) ref) = ref [] in
  let (curr_cu_infos:(frag list) ref) = ref [] in
  let (curr_cu_line:(frag list) ref) = ref [] in
  let (curr_cu_frame:(frag list) ref) = ref [] in

  let emit_die die = prepend curr_cu_infos die in
  let emit_null_die _ = emit_die (BYTE 0) in

  let dw_form_block1 (ops:dw_op array) : Asm.frag =
    let frag = SEQ (Array.map (dw_op_to_frag abi) ops) in
    let block_fixup = new_fixup "DW_FORM_block1 fixup" in
      SEQ [| WORD (TY_u8, F_SZ block_fixup);
             DEF (block_fixup, frag) |]
  in

  let dw_form_ref_addr (fix:fixup) : Asm.frag =
    WORD (signed_word_ty_mach,
          SUB ((M_POS fix), M_POS cu_info_section_fixup))
  in

  let encode_effect eff =
    (* Note: weird encoding: mutable+pure = unsafe. *)
    let mut_byte, pure_byte =
      match eff with
          Ast.UNSAFE -> (1,1)
        | Ast.STATE -> (1,0)
        | Ast.IO -> (0,0)
        | Ast.PURE -> (0,1)
    in
      SEQ [|
        (* DW_AT_mutable: DW_FORM_flag *)
        BYTE mut_byte;
        (* DW_AT_pure: DW_FORM_flag *)
        BYTE pure_byte;
      |]
  in

  (* Type-param DIEs. *)

  let type_param_die (p:(ty_param_idx * Ast.effect)) =
    let (idx, eff) = p in
      SEQ [|
        uleb (get_abbrev_code abbrev_rust_type_param);
        (* DW_AT_rust_type_code: DW_FORM_data1 *)
        BYTE (dw_rust_type_to_int DW_RUST_type_param);
        (* DW_AT_rust_type_param_index: DW_FORM_data4 *)
        WORD (word_ty_mach, IMM (Int64.of_int idx));
        encode_effect eff;
      |]
  in

  (* Type DIEs. *)

  let (emitted_types:(Ast.ty, Asm.frag) Hashtbl.t) = Hashtbl.create 0 in
  let (emitted_slots:(Ast.slot, Asm.frag) Hashtbl.t) = Hashtbl.create 0 in

  let rec ref_slot_die
      (slot:Ast.slot)
      : frag =
    if Hashtbl.mem emitted_slots slot
    then Hashtbl.find emitted_slots slot
    else
      let ref_addr_for_fix fix =
        let res = dw_form_ref_addr fix in
          Hashtbl.add emitted_slots slot res;
          res
      in

        match slot.Ast.slot_mode with
          | Ast.MODE_interior ->
              ref_type_die (slot_ty slot)

          | Ast.MODE_alias ->
              let fix = new_fixup "alias DIE" in
                emit_die (DEF (fix, SEQ [|
                                 uleb (get_abbrev_code abbrev_alias_slot);
                                 (* DW_AT_type: DW_FORM_ref_addr *)
                                 (ref_type_die (slot_ty slot));
                               |]));
                ref_addr_for_fix fix


  and size_block4 (sz:size) (add_to_base:bool) : frag =
    (* NB: typarams = "words following implicit args" by convention in
     * ABI/x86.
     *)
    let abi = cx.ctxt_abi in
    let typarams =
      Int64.add abi.Abi.abi_frame_base_sz abi.Abi.abi_implicit_args_sz
    in
    let word_n n = Int64.mul abi.Abi.abi_word_sz (Int64.of_int n) in
    let param_n n = Int64.add typarams (word_n n) in
    let param_n_field_k n k =
      [ DW_OP_fbreg (IMM (param_n n));
        DW_OP_deref;
        DW_OP_constu (IMM (word_n k));
        DW_OP_plus;
        DW_OP_deref ]
    in
    let rec sz_ops (sz:size) : dw_op list =
      match sz with
          SIZE_fixed i ->
            [ DW_OP_constu (IMM i) ]

        | SIZE_fixup_mem_sz fix ->
            [ DW_OP_constu (M_SZ fix) ]

        | SIZE_fixup_mem_pos fix ->
            [ DW_OP_constu (M_POS fix) ]

        | SIZE_param_size i ->
            param_n_field_k i Abi.tydesc_field_size

        | SIZE_param_align i ->
            param_n_field_k i Abi.tydesc_field_align

        | SIZE_rt_neg s ->
            (sz_ops s) @ [ DW_OP_neg ]

        | SIZE_rt_add (a, b) ->
            (sz_ops a) @ (sz_ops b) @ [ DW_OP_plus ]

        | SIZE_rt_mul (a, b) ->
            (sz_ops a) @ (sz_ops b) @ [ DW_OP_mul ]

        | SIZE_rt_max (a, b) ->
            (sz_ops a) @ (sz_ops b) @
              [ DW_OP_over;   (* ... a b a          *)
                DW_OP_over;   (* ... a b a b        *)
                DW_OP_ge;     (* ... a b (a>=b?1:0) *)

                (* jump +1 byte of dwarf ops if 1   *)
                DW_OP_bra (IMM 1L);

                (* do this if 0, when b is max.     *)
                DW_OP_swap;   (* ... b a            *)

                (* jump to here when a is max.      *)
                DW_OP_drop;   (* ... max            *)
              ]

        | SIZE_rt_align (align, off) ->
          (*
           * calculate off + pad where:
           *
           * pad = (align - (off mod align)) mod align
           *
           * In our case it's always a power of two, 
           * so we can just do:
           * 
           * mask = align-1
           * off += mask
           * off &= ~mask
           * 
           *)
            (sz_ops off) @ (sz_ops align) @
              [
                DW_OP_lit 1;          (* ... off align 1      *)
                DW_OP_minus;          (* ... off mask         *)
                DW_OP_dup;            (* ... off mask mask    *)
                DW_OP_not;            (* ... off mask ~mask   *)
                DW_OP_rot;            (* ... ~mask off mask   *)
                DW_OP_plus;           (* ... ~mask (off+mask) *)
                DW_OP_and;            (* ... aligned          *)
              ]
    in
    let ops = sz_ops sz in
    let ops =
      if add_to_base
      then ops @ [ DW_OP_plus ]
      else ops
    in
    let frag = SEQ (Array.map (dw_op_to_frag abi) (Array.of_list ops)) in
    let block_fixup = new_fixup "DW_FORM_block4 fixup" in
      SEQ [| WORD (TY_u32, F_SZ block_fixup);
             DEF (block_fixup, frag) |]


  and ref_type_die
      (ty:Ast.ty)
      : frag =
    (* Returns a DW_FORM_ref_addr to the type. *)
    if Hashtbl.mem emitted_types ty
    then Hashtbl.find emitted_types ty
    else
      let ref_addr_for_fix fix =
        let res = dw_form_ref_addr fix in
          Hashtbl.add emitted_types ty res;
          res
      in

      let record trec =
        let rty = referent_type abi (Ast.TY_rec trec) in
        let rty_sz = Il.referent_ty_size abi.Abi.abi_word_bits in
        let fix = new_fixup "record type DIE" in
        let die = DEF (fix, SEQ [|
                         uleb (get_abbrev_code abbrev_struct_type);
                         (* DW_AT_byte_size: DW_FORM_block4 *)
                         size_block4 (rty_sz rty) false
                       |]);
        in
        let rtys =
          match rty with
              Il.StructTy rtys -> rtys
            | _ -> bug () "record type became non-struct referent_ty"
        in
          emit_die die;
          Array.iteri
            begin
              fun i (ident, ty) ->
                emit_die (SEQ [|
                            uleb (get_abbrev_code abbrev_struct_type_member);
                            (* DW_AT_name: DW_FORM_string *)
                            ZSTRING ident;
                            (* DW_AT_type: DW_FORM_ref_addr *)
                            (ref_type_die ty);
                            (* DW_AT_data_member_location: DW_FORM_block4 *)
                            size_block4
                              (Il.get_element_offset word_bits rtys i)
                              true;
                            (* DW_AT_byte_size: DW_FORM_block4 *)
                            size_block4 (rty_sz rtys.(i)) false |]);
            end
            trec;
          emit_null_die ();
          ref_addr_for_fix fix
      in

      let tup ttup =
        record (Array.mapi (fun i s ->
                              ("_" ^ (string_of_int i), s))
                  ttup)
      in

      let string_type _ =
        (* 
         * Strings, like vecs, are &[rc,alloc,fill,data...] 
         *)
        let fix = new_fixup "string type DIE" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_string_type);
                 (* (DW_AT_byte_size, DW_FORM_block1); *)
                 dw_form_block1 [| DW_OP_push_object_address;
                                   DW_OP_deref;
                                   DW_OP_lit (word_sz_int * 2);
                                   DW_OP_plus; |];
                 (* (DW_AT_data_location, DW_FORM_block1); *)
                 dw_form_block1 [| DW_OP_push_object_address;
                                   DW_OP_deref;
                                   DW_OP_lit (word_sz_int * 3);
                                   DW_OP_plus |]
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let base (name, encoding, byte_size) =
        let fix = new_fixup ("base type DIE: " ^ name) in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_base_type);
                 (* DW_AT_name: DW_FORM_string *)
                 ZSTRING name;
                 (* DW_AT_encoding: DW_FORM_data1 *)
                 BYTE (dw_ate_to_int encoding);
                 (* DW_AT_byte_size: DW_FORM_data1 *)
                 BYTE byte_size
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let unspecified_anon_struct _ =
        let fix = new_fixup "unspecified-anon-struct DIE" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code
                         abbrev_unspecified_anon_structure_type);
                 (* DW_AT_declaration: DW_FORM_flag *)
                 BYTE 1;
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let unspecified_struct rust_ty =
        let fix = new_fixup "unspecified-struct DIE" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_unspecified_structure_type);
                 (* DW_AT_rust_type_code: DW_FORM_data1 *)
                 BYTE (dw_rust_type_to_int rust_ty);
                 (* DW_AT_declaration: DW_FORM_flag *)
                 BYTE 1;
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let rust_type_param (p:(ty_param_idx * Ast.effect)) =
        let fix = new_fixup "rust-type-param DIE" in
        let die = DEF (fix, type_param_die p) in
          emit_die die;
          ref_addr_for_fix fix
      in

      let unspecified_ptr_with_ref rust_ty ref_addr =
        let fix = new_fixup ("unspecified-pointer-type-with-ref DIE") in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_unspecified_pointer_type);
                 (* DW_AT_rust_type_code: DW_FORM_data1 *)
                 BYTE (dw_rust_type_to_int rust_ty);
                 (* DW_AT_declaration: DW_FORM_flag *)
                 BYTE 1;
                 (* DW_AT_type: DW_FORM_ref_addr *)
                 ref_addr
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let formal_type slot =
        let fix = new_fixup "formal type" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_formal_type);
                 (* DW_AT_type: DW_FORM_ref_addr *)
                 (ref_slot_die slot);
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let fn_type tfn =
        let (tsig, taux) = tfn in
        let fix = new_fixup "fn type" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_subroutine_type);
                 (* DW_AT_type: DW_FORM_ref_addr *)
                 (ref_slot_die tsig.Ast.sig_output_slot);
                 encode_effect taux.Ast.fn_effect;
                 (* DW_AT_rust_iterator: DW_FORM_flag *)
                 BYTE (if taux.Ast.fn_is_iter then 1 else 0)
               |])
        in
          emit_die die;
          Array.iter
            (fun s -> ignore (formal_type s))
            tsig.Ast.sig_input_slots;
          emit_null_die ();
          ref_addr_for_fix fix
      in

      let obj_fn_type ident tfn =
        let (tsig, taux) = tfn in
        let fix = new_fixup "fn type" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_obj_subroutine_type);
                 (* DW_AT_name: DW_FORM_string *)
                 ZSTRING ident;
                 (* DW_AT_type: DW_FORM_ref_addr *)
                 (ref_slot_die tsig.Ast.sig_output_slot);
                 encode_effect taux.Ast.fn_effect;
                 (* DW_AT_rust_iterator: DW_FORM_flag *)
                 BYTE (if taux.Ast.fn_is_iter then 1 else 0)
               |])
        in
          emit_die die;
          Array.iter
            (fun s -> ignore (formal_type s))
            tsig.Ast.sig_input_slots;
          emit_null_die ();
          ref_addr_for_fix fix
      in

      let obj_type (eff,ob) =
        let fix = new_fixup "object type" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_obj_type);
                 encode_effect eff;
               |])
        in
          emit_die die;
          Hashtbl.iter (fun k v -> ignore (obj_fn_type k v)) ob;
          emit_null_die ();
          ref_addr_for_fix fix
      in

      let unspecified_ptr_with_ref_ty rust_ty ty =
        unspecified_ptr_with_ref rust_ty (ref_type_die ty)
      in

      let unspecified_ptr rust_ty =
        unspecified_ptr_with_ref rust_ty (unspecified_anon_struct ())
      in

      let native_ptr_type oid =
        let fix = new_fixup "native type" in
        let die =
          DEF (fix, SEQ [|
                 uleb (get_abbrev_code abbrev_native_pointer_type);
                 (* DW_AT_rust_type_code: DW_FORM_data1 *)
                 BYTE (dw_rust_type_to_int DW_RUST_native);
                 (* DW_AT_rust_native_type_id: DW_FORM_data4 *)
                 WORD (word_ty_mach, IMM (Int64.of_int (int_of_opaque oid)));
               |])
        in
          emit_die die;
          ref_addr_for_fix fix
      in

      let tag_type fix_opt ttag =
        (*
         * Tag-encoding is a bit complex. It's based on the pascal model.
         *
         * You have a structure (DW_TAG_structure_type) with 2 fields:
         *
         * 0 : the discriminant (type uint)
         * 1 : the variant-part of the structure (DW_TAG_variant_part)
         *     with DW_AT_discr pointing to the disctiminant, and kids:
         *     0 : variant 0 (DW_TAG_variant) with DW_AT_discr_value 0
         *         (with a tuple-type child)
         *     1 : variant 1 ...
         *     ...
         *     N : variant N (DW_TAG_variant) with DW_AT_discr_value N
         *
         * Curiously, DW_TAG_union_type doesn't seem to play into it.
         * I'm a bit surprised by that!
         *)

        let rty = referent_type abi (Ast.TY_tag ttag) in
        let rty_sz = Il.referent_ty_size abi.Abi.abi_word_bits in
        let rtys =
          match rty with
              Il.StructTy rtys -> rtys
            | _ -> bug () "tag type became non-struct referent_ty"
        in

        let outer_structure_fix =
          match fix_opt with
              None -> new_fixup "tag type"
            | Some f -> f
        in
        let outer_structure_die =
          DEF (outer_structure_fix, SEQ [|
                 uleb (get_abbrev_code abbrev_struct_type);
                 (* DW_AT_byte_size: DW_FORM_block4 *)
                 size_block4 (rty_sz rty) false
               |])
        in

        let discr_fix = new_fixup "tag discriminant" in
        let discr_die =
          DEF (discr_fix, SEQ [|
                 uleb (get_abbrev_code abbrev_struct_type_member);
                 (* DW_AT_name: DW_FORM_string *)
                 ZSTRING "tag";
                 (* DW_AT_type: DW_FORM_ref_addr *)
                 (ref_type_die Ast.TY_uint);
                 (* DW_AT_data_member_location: DW_FORM_block4 *)
                 size_block4
                   (Il.get_element_offset word_bits rtys 0)
                   true;
                 (* DW_AT_byte_size: DW_FORM_block4 *)
                 size_block4 (rty_sz rtys.(0)) false |]);
        in

        let variant_part_die =
          SEQ [|
            uleb (get_abbrev_code abbrev_variant_part);
            (* DW_AT_discr: DW_FORM_ref_addr *)
            (dw_form_ref_addr discr_fix)
          |]
        in

        let emit_variant i (*name*)_ ttup =
          (* FIXME: Possibly use a DW_TAG_enumeration_type here? *)
          (* Tag-names aren't getting encoded; I'm not sure if that's a
           * problem. Might be. *)
          emit_die (SEQ [|
                      uleb (get_abbrev_code abbrev_variant);
                      (* DW_AT_discr_value: DW_FORM_udata *)
                      uleb i;
                    |]);
          ignore (tup ttup);
          emit_null_die ();
        in
          emit_die outer_structure_die;
          emit_die discr_die;
          emit_die variant_part_die;
          let tag_keys = sorted_htab_keys ttag in
            Array.iteri
              (fun i k -> emit_variant i k (Hashtbl.find ttag k))
              tag_keys;
            emit_null_die (); (* end variant-part *)
            emit_null_die (); (* end outer struct *)
            ref_addr_for_fix outer_structure_fix
      in

      let iso_type tiso =
        let iso_fixups =
          Array.map
            (fun _ -> new_fixup "iso-member tag type")
            tiso.Ast.iso_group
        in
          Stack.push iso_fixups iso_stack;
          let tag_dies =
            Array.mapi
              (fun i fix ->
                 tag_type (Some fix) tiso.Ast.iso_group.(i))
              iso_fixups
          in
            ignore (Stack.pop iso_stack);
            tag_dies.(tiso.Ast.iso_index)
      in

      let idx_type i =
        ref_addr_for_fix (Stack.top iso_stack).(i)
      in

      let exterior_type t =
        let fix = new_fixup "exterior DIE" in
        let body_off =
          word_sz_int * Abi.exterior_rc_slot_field_body
        in
          emit_die (DEF (fix, SEQ [|
                           uleb (get_abbrev_code abbrev_exterior_type);
                           (* DW_AT_type: DW_FORM_ref_addr *)
                           (ref_type_die t);
                           (* DW_AT_data_location: DW_FORM_block1 *)
                           (* This is a DWARF expression for moving
                              from the address of an exterior
                              allocation to the address of its
                              body. *)
                           dw_form_block1
                             [| DW_OP_push_object_address;
                                DW_OP_lit body_off;
                                DW_OP_plus;
                                DW_OP_deref |]
                         |]));
          ref_addr_for_fix fix
      in

      let mutable_type t =
        let fix = new_fixup "mutable DIE" in
          emit_die (DEF (fix, SEQ [|
                           uleb (get_abbrev_code abbrev_mutable_type);
                           (* DW_AT_type: DW_FORM_ref_addr *)
                           (ref_type_die t);
                           (* DW_AT_mutable: DW_FORM_flag *)
                           BYTE 1;
                         |]));
          ref_addr_for_fix fix
      in

        match ty with
            Ast.TY_nil -> unspecified_struct DW_RUST_nil
          | Ast.TY_bool -> base ("bool", DW_ATE_boolean, 1)
          | Ast.TY_mach (TY_u8)  -> base ("u8",  DW_ATE_unsigned, 1)
          | Ast.TY_mach (TY_u16) -> base ("u16", DW_ATE_unsigned, 2)
          | Ast.TY_mach (TY_u32) -> base ("u32", DW_ATE_unsigned, 4)
          | Ast.TY_mach (TY_u64) -> base ("u64", DW_ATE_unsigned, 8)
          | Ast.TY_mach (TY_i8)  -> base ("i8",  DW_ATE_signed, 1)
          | Ast.TY_mach (TY_i16) -> base ("i16", DW_ATE_signed, 2)
          | Ast.TY_mach (TY_i32) -> base ("i32", DW_ATE_signed, 4)
          | Ast.TY_mach (TY_i64) -> base ("i64", DW_ATE_signed, 8)
          | Ast.TY_int -> base ("int", DW_ATE_signed, word_sz_int)
          | Ast.TY_uint -> base ("uint", DW_ATE_unsigned, word_sz_int)
          | Ast.TY_char -> base ("char", DW_ATE_unsigned_char, 4)
          | Ast.TY_str -> string_type ()
          | Ast.TY_rec trec -> record trec
          | Ast.TY_tup ttup -> tup ttup
          | Ast.TY_tag ttag -> tag_type None ttag
          | Ast.TY_iso tiso -> iso_type tiso
          | Ast.TY_idx i -> idx_type i
          | Ast.TY_vec t -> unspecified_ptr_with_ref_ty DW_RUST_vec t
          | Ast.TY_chan t -> unspecified_ptr_with_ref_ty DW_RUST_chan t
          | Ast.TY_port t -> unspecified_ptr_with_ref_ty DW_RUST_port t
          | Ast.TY_task -> unspecified_ptr DW_RUST_task
          | Ast.TY_fn fn -> fn_type fn
          | Ast.TY_type -> unspecified_ptr DW_RUST_type
          | Ast.TY_native i -> native_ptr_type i
          | Ast.TY_param p -> rust_type_param p
          | Ast.TY_obj ob -> obj_type ob
          | Ast.TY_mutable t -> mutable_type t
          | Ast.TY_exterior t -> exterior_type t
          | _ ->
              bug () "unimplemented dwarf encoding for type %a"
                Ast.sprintf_ty ty
  in

  let finish_crate_cu_and_compose_headers _ =

    let pubnames_header_and_curr_pubnames =
      SEQ [| (BYTE 0) |]
    in

    let aranges_header_and_curr_aranges =
      SEQ [| (BYTE 0) |]
    in

    let cu_info_fixup = new_fixup "CU debug_info fixup" in
    let info_header_fixup = new_fixup "CU debug_info header" in
    let info_header_and_curr_infos =
      SEQ
        [|
          WORD (TY_u32,                          (* unit_length:          *)
                (ADD
                   ((F_SZ cu_info_fixup),        (* including this header,*)
                    (F_SZ info_header_fixup)))); (* excluding this word.  *)
          DEF (info_header_fixup,
               (SEQ [|
                  WORD (TY_u16, IMM 2L);         (* DWARF version         *)
                  (* Since we share abbrevs across all CUs, 
                   * offset is always 0.
                   *)
                  WORD (TY_u32, IMM 0L);         (* CU-abbrev offset.     *)
                  BYTE 4;                        (* Size of an address.   *)
                |]));
          DEF (cu_info_fixup,
               SEQ (Array.of_list (List.rev (!curr_cu_infos))));
        |]
    in

    let cu_line_fixup = new_fixup "CU debug_line fixup" in
    let cu_line_header_fixup = new_fixup "CU debug_line header" in
    let line_header_fixup = new_fixup "CU debug_line header" in
    let line_header_and_curr_line =
      SEQ
        [|
          WORD
            (TY_u32,                              (* unit_length:         *)
             (ADD
                ((F_SZ cu_line_fixup),           (* including this header,*)
                 (F_SZ cu_line_header_fixup)))); (* excluding this word.  *)
          DEF (cu_line_header_fixup,
               (SEQ [|
                  WORD (TY_u16, IMM 2L);         (* DWARF version.        *)
                  WORD
                    (TY_u32,
                     (F_SZ line_header_fixup));  (* Another header-length.*)
                  DEF (line_header_fixup,
                       SEQ [|
                         BYTE 1;                 (* Minimum insn length.  *)
                         BYTE 1;                 (* default_is_stmt       *)
                         BYTE 0;                 (* line_base             *)
                         BYTE 0;                 (* line_range            *)
                         BYTE (max_dw_lns + 1);  (* opcode_base           *)
                         BYTES                   (* opcode arity array.   *)
                           (Array.init max_dw_lns
                              (fun i ->
                                 (dw_lns_arity
                                    (int_to_dw_lns
                                       (i+1)))));
                         (BYTE 0);               (* List of include dirs. *)
                         (BYTE 0);               (* List of file entries. *)
                       |])|]));
          DEF (cu_line_fixup,
               SEQ (Array.of_list (List.rev (!curr_cu_line))));
        |]
    in
    let frame_header_and_curr_frame =
      SEQ [| (BYTE 0) |]
    in
    let prepend_and_reset (curr_ref, accum_ref, header_and_curr) =
      prepend accum_ref header_and_curr;
      curr_ref := []
    in
      List.iter prepend_and_reset
        [ (curr_cu_aranges, cu_aranges, aranges_header_and_curr_aranges);
          (curr_cu_pubnames, cu_pubnames, pubnames_header_and_curr_pubnames);
          (curr_cu_infos, cu_infos, info_header_and_curr_infos);
          (curr_cu_line, cu_lines, line_header_and_curr_line);
          (curr_cu_frame, cu_frames, frame_header_and_curr_frame) ]
  in

  let image_base_rel (fix:fixup) : expr64 =
    SUB (M_POS (fix), M_POS (cx.ctxt_image_base_fixup))
  in

  let addr_ranges (fix:fixup) : frag =
    let image_is_relocated =
      match cx.ctxt_sess.Session.sess_targ with
          Win32_x86_pe ->
            cx.ctxt_sess.Session.sess_library_mode
        | _ -> true
    in
    let lo =
      if image_is_relocated
      then image_base_rel fix
      else M_POS fix
    in
      SEQ [|
        (* DW_AT_low_pc, DW_FORM_addr *)
        WORD (word_ty_mach, lo);
        (* DW_AT_high_pc, DW_FORM_addr *)
        WORD (word_ty_mach, ADD ((lo),
                                 (M_SZ fix)))
      |]
  in

  let emit_srcfile_cu_die
      (name:string)
      (cu_text_fixup:fixup)
      : unit =
    let abbrev_code = get_abbrev_code abbrev_srcfile_cu in
    let srcfile_cu_die =
      (SEQ [|
         uleb abbrev_code;
         (* DW_AT_name:  DW_FORM_string *)
         ZSTRING (Filename.basename name);
         (* DW_AT_comp_dir:  DW_FORM_string *)
         ZSTRING (Filename.concat (Sys.getcwd()) (Filename.dirname name));
         addr_ranges cu_text_fixup;
       |])
    in
      emit_die srcfile_cu_die
  in

  let emit_meta_die
      (meta:(Ast.ident * string))
      : unit =
    let abbrev_code = get_abbrev_code abbrev_meta in
    let die =
      SEQ [| uleb abbrev_code;
             (* DW_AT_name: DW_FORM_string *)
             ZSTRING (fst meta);
             (* DW_AT_const_value: DW_FORM_string *)
             ZSTRING (snd meta);
          |]
    in
      emit_die die
  in

  let begin_crate_cu_and_emit_cu_die
      (name:string)

      (cu_text_fixup:fixup)
      : unit =
    let abbrev_code = get_abbrev_code abbrev_crate_cu in
    let crate_cu_die =
      (SEQ [|
         uleb abbrev_code;
         (* DW_AT_producer:  DW_FORM_string *)
         ZSTRING "Rustboot pre-release";
         (* DW_AT_language:  DW_FORM_data4 *)
         WORD (word_ty_mach, IMM 0x2L);     (* DW_LANG_C *)
         (* DW_AT_name:  DW_FORM_string *)
         ZSTRING (Filename.basename name);
         (* DW_AT_comp_dir:  DW_FORM_string *)
         ZSTRING (Filename.concat (Sys.getcwd()) (Filename.dirname name));
         addr_ranges cu_text_fixup;
         (* DW_AT_use_UTF8, DW_FORM_flag *)
         BYTE 1
       |])
    in
      curr_cu_infos := [crate_cu_die];
      curr_cu_line := []
  in

  let type_param_decl_die (p:(Ast.ident * (ty_param_idx * Ast.effect))) =
    let (ident, (idx, eff)) = p in
      SEQ [|
        uleb (get_abbrev_code abbrev_rust_type_param_decl);
        (* DW_AT_rust_type_code: DW_FORM_data1 *)
        BYTE (dw_rust_type_to_int DW_RUST_type_param);
        (* DW_AT_name:  DW_FORM_string *)
        ZSTRING (Filename.basename ident);
        (* DW_AT_rust_type_param_index: DW_FORM_data4 *)
        WORD (word_ty_mach, IMM (Int64.of_int idx));
        encode_effect eff;
      |]
  in

  let emit_type_param_decl_dies
      (params:(Ast.ty_param identified) array)
      : unit =
    Array.iter
      (fun p ->
         emit_die (type_param_decl_die p.node))
      params;
  in

  let emit_module_die
      (id:Ast.ident)
      : unit =
    let abbrev_code = get_abbrev_code abbrev_module in
    let module_die =
      (SEQ [|
         uleb abbrev_code;
         (* DW_AT_name *)
         ZSTRING id;
       |])
    in
      emit_die module_die;
  in

  let emit_subprogram_die
      (id:Ast.ident)
      (ret_slot:Ast.slot)
      (effect:Ast.effect)
      (fix:fixup)
      : unit =
    (* NB: retpc = "top word of frame-base" by convention in ABI/x86. *)
    let abi = cx.ctxt_abi in
    let retpc = Int64.sub abi.Abi.abi_frame_base_sz abi.Abi.abi_word_sz in
    let abbrev_code = get_abbrev_code abbrev_subprogram in
    let subprogram_die =
      (SEQ [|
         uleb abbrev_code;
         (* DW_AT_name *)
         ZSTRING id;
         (* DW_AT_type: DW_FORM_ref_addr *)
         ref_slot_die ret_slot;
         addr_ranges fix;
         (* DW_AT_frame_base *)
         dw_form_block1 [| DW_OP_reg abi.Abi.abi_dwarf_fp_reg |];
         (* DW_AT_return_addr *)
         dw_form_block1 [| DW_OP_fbreg (Asm.IMM retpc); |];
         encode_effect effect;
       |])
    in
      emit_die subprogram_die
  in

  let emit_typedef_die
      (id:Ast.ident)
      (ty:Ast.ty)
      : unit =
    let abbrev_code = get_abbrev_code abbrev_typedef in
    let typedef_die =
      (SEQ [|
         uleb abbrev_code;
         (* DW_AT_name: DW_FORM_string *)
         ZSTRING id;
         (* DW_AT_type: DW_FORM_ref_addr *)
         (ref_type_die ty);
       |])
    in
      emit_die typedef_die
  in

  let visit_crate_pre
      (crate:Ast.crate)
      : unit =
    let filename = (Hashtbl.find cx.ctxt_item_files crate.id) in
      log cx "walking crate CU '%s'" filename;
      begin_crate_cu_and_emit_cu_die filename
        (Hashtbl.find cx.ctxt_file_fixups crate.id);
      Array.iter emit_meta_die crate.node.Ast.crate_meta;
      inner.Walk.visit_crate_pre crate
  in

  let visit_mod_item_pre
      (id:Ast.ident)
      (params:(Ast.ty_param identified) array)
      (item:Ast.mod_item)
      : unit =
    if Hashtbl.mem cx.ctxt_item_files item.id
    then
      begin
        let filename = (Hashtbl.find cx.ctxt_item_files item.id) in
          log cx "walking srcfile CU '%s'" filename;
          emit_srcfile_cu_die filename
            (Hashtbl.find cx.ctxt_file_fixups item.id);
      end
    else
      ();
    begin
      match item.node.Ast.decl_item with
          Ast.MOD_ITEM_mod _ ->
            begin
              log cx "walking module '%s' with %d type params"
                (path_name())
                (Array.length item.node.Ast.decl_params);
              emit_module_die id;
              emit_type_param_decl_dies item.node.Ast.decl_params;
            end
        | Ast.MOD_ITEM_fn _ ->
            begin
              let ty = Hashtbl.find cx.ctxt_all_item_types item.id in
              let (tsig,taux) =
                match ty with
                    Ast.TY_fn tfn -> tfn
                  | _ ->
                      bug ()
                        "non-fn type when emitting dwarf for MOD_ITEM_fn"
              in
                log cx "walking function '%s' with %d type params"
                  (path_name())
                  (Array.length item.node.Ast.decl_params);
                emit_subprogram_die
                  id tsig.Ast.sig_output_slot taux.Ast.fn_effect
                  (Hashtbl.find cx.ctxt_fn_fixups item.id);
                emit_type_param_decl_dies item.node.Ast.decl_params;
            end
        | Ast.MOD_ITEM_type _ ->
            begin
              log cx "walking typedef '%s' with %d type params"
                (path_name())
                (Array.length item.node.Ast.decl_params);
              emit_typedef_die
                id (Hashtbl.find cx.ctxt_all_type_items item.id);
              emit_type_param_decl_dies item.node.Ast.decl_params;
            end
        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre id params item
  in

  let visit_crate_post
      (crate:Ast.crate)
      : unit =
    inner.Walk.visit_crate_post crate;
    assert (Hashtbl.mem cx.ctxt_item_files crate.id);
    emit_null_die();
    log cx
      "finishing crate CU and composing headers (%d DIEs collected)"
      (List.length (!curr_cu_infos));
    finish_crate_cu_and_compose_headers ()
  in

  let visit_mod_item_post
      (id:Ast.ident)
      (params:(Ast.ty_param identified) array)
      (item:Ast.mod_item)
      : unit =
    inner.Walk.visit_mod_item_post id params item;
    begin
      match item.node.Ast.decl_item with
          Ast.MOD_ITEM_mod _
        | Ast.MOD_ITEM_fn _
        | Ast.MOD_ITEM_type _ -> emit_null_die ()
        | _ -> ()
    end;
    if Hashtbl.mem cx.ctxt_item_files item.id
    then emit_null_die()
  in

  let visit_block_pre (b:Ast.block) : unit =
    log cx "entering lexical block";
    let fix = Hashtbl.find cx.ctxt_block_fixups b.id in
    let abbrev_code = get_abbrev_code abbrev_lexical_block in
    let block_die =
      SEQ [|
        uleb abbrev_code;
        addr_ranges fix;
      |]
    in
      emit_die block_die;
      inner.Walk.visit_block_pre b
  in

  let visit_block_post (b:Ast.block) : unit =
    inner.Walk.visit_block_post b;
    log cx "leaving lexical block, terminating with NULL DIE";
    emit_null_die ()
  in

  let visit_slot_identified_pre (s:Ast.slot identified) : unit =
    begin
      match htab_search cx.ctxt_slot_keys s.id with
          None
        | Some Ast.KEY_temp _ -> ()
        | Some Ast.KEY_ident ident ->
            begin
              let abbrev_code =
                if Hashtbl.mem cx.ctxt_slot_is_arg s.id
                then get_abbrev_code abbrev_formal
                else get_abbrev_code abbrev_variable
              in
              let resolved_slot = referent_to_slot cx s.id in
              let emit_var_die slot_loc =
                let var_die =
                  SEQ [|
                    uleb abbrev_code;
                    (* DW_AT_name: DW_FORM_string *)
                    ZSTRING ident;
                    (* DW_AT_location:  DW_FORM_block1 *)
                    dw_form_block1 slot_loc;
                    (* DW_AT_type: DW_FORM_ref_addr *)
                    ref_slot_die resolved_slot
                  |]
                in
                  emit_die var_die;
              in
                match htab_search cx.ctxt_slot_offsets s.id with
                    Some off ->
                      begin
                        match Il.size_to_expr64 off with
                            (* FIXME (issue #73): handle dynamic-size
                             * slots.
                             *)
                            None -> ()
                          | Some off ->
                              emit_var_die
                                [| DW_OP_fbreg off |]
                      end
                  | None ->
                      (* FIXME (issue #28): handle slots assigned to
                       * vregs.
                       *)
                      ()
            end
    end;
    inner.Walk.visit_slot_identified_pre s
  in

    { inner with
        Walk.visit_crate_pre = visit_crate_pre;
        Walk.visit_crate_post = visit_crate_post;
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_block_pre = visit_block_pre;
        Walk.visit_block_post = visit_block_post;
        Walk.visit_slot_identified_pre = visit_slot_identified_pre
    }
;;


let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : debug_records =

  let cu_aranges = ref [] in
  let cu_pubnames = ref [] in
  let cu_infos = ref [] in
  let cu_abbrevs = ref [] in
  let cu_lines = ref [] in
  let cu_frames = ref [] in

  let path = Stack.create () in

  let passes =
    [|
      unreferenced_required_item_ignoring_visitor cx
        (dwarf_visitor cx Walk.empty_visitor path
           cx.ctxt_debug_info_fixup
           cu_aranges cu_pubnames
           cu_infos cu_abbrevs
           cu_lines cu_frames)
    |];
  in

    log cx "emitting DWARF records";
    run_passes cx "dwarf" path passes (log cx "%s") crate;

    (* Terminate the tables. *)
    {
      debug_aranges = SEQ (Array.of_list (List.rev (!cu_aranges)));
      debug_pubnames = SEQ (Array.of_list (List.rev (!cu_pubnames)));
      debug_info = SEQ (Array.of_list (List.rev (!cu_infos)));
      debug_abbrev = SEQ (Array.of_list (List.rev (!cu_abbrevs)));
      debug_line = SEQ (Array.of_list (List.rev (!cu_lines)));
      debug_frame = SEQ (Array.of_list (List.rev (!cu_frames)));
    }
;;

(*
 * Support for reconstituting a DWARF tree from a file, and various
 * artifacts we can distill back from said DWARF.
 *)

let log sess = Session.log "dwarf"
  sess.Session.sess_log_dwarf
  sess.Session.sess_log_out
;;


let iflog (sess:Session.sess) (thunk:(unit -> unit)) : unit =
  if sess.Session.sess_log_dwarf
  then thunk ()
  else ()
;;

let read_abbrevs
    (sess:Session.sess)
    (ar:asm_reader)
    ((off:int),(sz:int))
    : (int,abbrev) Hashtbl.t =
  ar.asm_seek off;
  let abs = Hashtbl.create 0 in
  let rec read_abbrevs _ =
    if ar.asm_get_off() >= (off + sz)
    then abs
    else
      begin
        let n = ar.asm_get_uleb() in
        let tag = ar.asm_get_uleb() in
        let has_children = ar.asm_get_u8() in
        let pairs = ref [] in
        let _ =
          log sess "abbrev: %d, tag: %d, has_children: %d"
            n tag has_children
        in
        let rec read_pairs _ =
          let attr = ar.asm_get_uleb() in
          let form = ar.asm_get_uleb() in
          let _ = log sess "attr: %d, form: %d" attr form in
            match (attr,form) with
                (0,0) -> Array.of_list (List.rev (!pairs))
              | _ ->
                  begin
                    pairs := (dw_at_of_int attr,
                              dw_form_of_int form) :: (!pairs);
                    read_pairs()
                  end
        in
        let pairs = read_pairs() in
          Hashtbl.add abs n (dw_tag_of_int tag,
                             dw_children_of_int has_children,
                             pairs);
          read_abbrevs()
      end;
  in
    read_abbrevs()
;;

type data =
    DATA_str of string
  | DATA_num of int
  | DATA_other
;;

type die =
    { die_off: int;
      die_tag: dw_tag;
      die_attrs: (dw_at * (dw_form * data)) array;
      die_children: die array; }
;;

type rooted_dies = (int * (int,die) Hashtbl.t)
;;

let fmt_dies
    (ff:Format.formatter)
    (dies:rooted_dies)
    : unit =
  let ((root:int),(dies:(int,die) Hashtbl.t)) = dies in
  let rec fmt_die die =
    Fmt.fmt ff "@\nDIE <0x%x> %s" die.die_off (dw_tag_to_string die.die_tag);
    Array.iter
      begin
        fun (at,(form,data)) ->
          Fmt.fmt ff "@\n  %s = " (dw_at_to_string at);
          begin
            match data with
                DATA_num n -> Fmt.fmt ff "0x%x"  n
              | DATA_str s -> Fmt.fmt ff "\"%s\"" s
              | DATA_other -> Fmt.fmt ff "<other>"
          end;
          Fmt.fmt ff "  (%s)" (dw_form_to_string form)
      end
      die.die_attrs;
    if (Array.length die.die_children) != 0
    then
      begin
        Fmt.fmt ff "@\n";
        Fmt.fmt_obox ff;
        Fmt.fmt ff "  children: ";
        Fmt.fmt_obr ff;
        Array.iter fmt_die die.die_children;
        Fmt.fmt_cbb ff
      end;
  in
    fmt_die (Hashtbl.find dies root)
;;

let read_dies
    (sess:Session.sess)
    (ar:asm_reader)
    ((off:int),(sz:int))
    (abbrevs:(int,abbrev) Hashtbl.t)
    : (int * ((int,die) Hashtbl.t)) =
  ar.asm_seek off;
  let cu_len = ar.asm_get_u32() in
  let _ = log sess "debug_info cu_len: %d, section size %d" cu_len sz in
  let _ = assert ((cu_len + 4) = sz) in
  let dwarf_vers = ar.asm_get_u16() in
  let _ = assert (dwarf_vers >= 2) in
  let cu_abbrev_off = ar.asm_get_u32() in
  let _ = assert (cu_abbrev_off = 0) in
  let sizeof_addr = ar.asm_get_u8() in
  let _ = assert (sizeof_addr = 4) in

  let adv_block1 _ =
    let len = ar.asm_get_u8() in
      ar.asm_adv len
  in

  let adv_block4 _ =
    let len = ar.asm_get_u32() in
      ar.asm_adv len
  in

  let all_dies = Hashtbl.create 0 in
  let root = (ar.asm_get_off()) - off in

  let rec read_dies (dies:(die list) ref) =
    let die_arr _ = Array.of_list (List.rev (!dies)) in
      if ar.asm_get_off() >= (off + sz)
      then die_arr()
      else
        begin
          let die_off = (ar.asm_get_off()) - off in
          let abbrev_num = ar.asm_get_uleb() in
            if abbrev_num = 0
            then die_arr()
            else
              let _ =
                log sess "DIE at off <%d> with abbrev %d"
                  die_off abbrev_num
              in
              let abbrev = Hashtbl.find abbrevs abbrev_num in
              let (tag, children, attrs) = abbrev in
              let attrs =
                Array.map
                  begin
                    fun (attr,form) ->
                      let data =
                        match form with
                            DW_FORM_string -> DATA_str (ar.asm_get_zstr())
                          | DW_FORM_addr -> DATA_num (ar.asm_get_u32())
                          | DW_FORM_ref_addr -> DATA_num (ar.asm_get_u32())
                          | DW_FORM_data1 -> DATA_num (ar.asm_get_u8())
                          | DW_FORM_data4 -> DATA_num (ar.asm_get_u32())
                          | DW_FORM_flag -> DATA_num (ar.asm_get_u8())
                          | DW_FORM_block1 -> (adv_block1(); DATA_other)
                          | DW_FORM_block4 -> (adv_block4(); DATA_other)
                          | _ ->
                              bug () "unknown DWARF form %d"
                                (dw_form_to_int form)
                      in
                        (attr, (form, data))
                  end
                  attrs;
              in
              let children =
                match children with
                    DW_CHILDREN_yes -> read_dies (ref [])
                  | DW_CHILDREN_no -> [| |]
              in
              let die = { die_off = die_off;
                          die_tag = tag;
                          die_attrs = attrs;
                          die_children = children }
              in
                prepend dies die;
                htab_put all_dies die_off die;
                read_dies dies
        end
  in
    ignore (read_dies (ref []));
    iflog sess
      begin
        fun _ ->
          log sess "read DIEs:";
          log sess "%s" (Fmt.fmt_to_str fmt_dies (root, all_dies));
      end;
    (root, all_dies)
;;

let rec extract_meta
    ((i:int),(dies:(int,die) Hashtbl.t))
    :  (Ast.ident * string) array =
  let meta = Queue.create () in

  let get_attr die attr =
    atab_find die.die_attrs attr
  in

  let get_str die attr  =
    match get_attr die attr with
        (_, DATA_str s) -> s
      | _ -> bug () "unexpected num form for %s" (dw_at_to_string attr)
  in

  let die = Hashtbl.find dies i in
    begin
      match die.die_tag with
          DW_TAG_rust_meta ->
            let n = get_str die DW_AT_name in
            let v = get_str die DW_AT_const_value in
              Queue.add (n,v) meta

        | DW_TAG_compile_unit ->
            Array.iter
              (fun child ->
                 Array.iter (fun m -> Queue.add m meta)
                   (extract_meta (child.die_off,dies)))
              die.die_children

        | _ -> ()
    end;
    queue_to_arr meta
;;


let rec extract_mod_items
    (nref:node_id ref)
    (oref:opaque_id ref)
    (abi:Abi.abi)
    (mis:Ast.mod_items)
    ((i:int),(dies:(int,die) Hashtbl.t))
    : unit =

  let next_node_id _ : node_id =
    let id = !nref in
      nref:= Node ((int_of_node id)+1);
      id
  in

  let next_opaque_id _ : opaque_id =
    let id = !oref in
      oref:= Opaque ((int_of_opaque id)+1);
      id
  in

  let external_opaques = Hashtbl.create 0 in
  let get_opaque_of o =
    htab_search_or_add external_opaques o
      (fun _ -> next_opaque_id())
  in


  let (word_sz:int64) = abi.Abi.abi_word_sz in
  let (word_sz_int:int) = Int64.to_int word_sz in

  let get_die i =
    Hashtbl.find dies i
  in

  let get_attr die attr =
    atab_find die.die_attrs attr
  in

  let get_str die attr  =
    match get_attr die attr with
        (_, DATA_str s) -> s
      | _ -> bug () "unexpected num form for %s" (dw_at_to_string attr)
  in

  let get_num die attr =
    match get_attr die attr with
        (_, DATA_num n) -> n
      | _ -> bug () "unexpected str form for %s" (dw_at_to_string attr)
  in

  let get_flag die attr =
    match get_attr die attr with
        (_, DATA_num 0) -> false
      | (_, DATA_num 1) -> true
      | _ -> bug () "unexpected non-flag form for %s" (dw_at_to_string attr)
  in

  let get_effect die =
    match (get_flag die DW_AT_mutable, get_flag die DW_AT_pure) with
        (* Note: weird encoding: mutable+pure = unsafe. *)
        (true, true) -> Ast.UNSAFE
      | (true, false) -> Ast.STATE
      | (false, false) -> Ast.IO
      | (false, true) -> Ast.PURE
  in

  let get_name die = get_str die DW_AT_name in

  let get_type_param die =
    let idx = get_num die DW_AT_rust_type_param_index in
    let e = get_effect die in
      (idx, e)
  in

  let get_native_id die =
    get_num die DW_AT_rust_native_type_id
  in

  let get_type_param_decl die =
    ((get_str die DW_AT_name), (get_type_param die))
  in

  let is_rust_type die t =
    match atab_search die.die_attrs DW_AT_rust_type_code with
        Some (_, DATA_num n) -> (dw_rust_type_of_int n) = t
      | _ -> false
  in

  let rec get_ty die : Ast.ty =
      match die.die_tag with

          DW_TAG_structure_type
            when is_rust_type die DW_RUST_nil ->
              Ast.TY_nil

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_task ->
            Ast.TY_task

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_type ->
            Ast.TY_type

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_port ->
            Ast.TY_port (get_referenced_ty die)

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_chan ->
            Ast.TY_chan (get_referenced_ty die)

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_vec ->
            Ast.TY_vec (get_referenced_ty die)

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_type_param ->
            Ast.TY_param (get_type_param die)

        | DW_TAG_pointer_type
            when is_rust_type die DW_RUST_native ->
            Ast.TY_native (get_opaque_of (get_native_id die))

        | DW_TAG_pointer_type ->
            Ast.TY_exterior (get_referenced_ty die)

        | DW_TAG_const_type
            when ((get_num die DW_AT_mutable) = 1) ->
            Ast.TY_mutable (get_referenced_ty die)

        | DW_TAG_string_type -> Ast.TY_str

        | DW_TAG_base_type ->
            begin
              match ((get_name die),
                     (dw_ate_of_int (get_num die DW_AT_encoding)),
                     (get_num die DW_AT_byte_size)) with
                  ("bool", DW_ATE_boolean, 1) -> Ast.TY_bool
                | ("u8", DW_ATE_unsigned, 1) -> Ast.TY_mach TY_u8
                | ("u16", DW_ATE_unsigned, 2) -> Ast.TY_mach TY_u16
                | ("u32", DW_ATE_unsigned, 4) -> Ast.TY_mach TY_u32
                | ("u64", DW_ATE_unsigned, 8) -> Ast.TY_mach TY_u64
                | ("i8", DW_ATE_signed, 1) -> Ast.TY_mach TY_i8
                | ("i16", DW_ATE_signed, 2) -> Ast.TY_mach TY_i16
                | ("i32", DW_ATE_signed, 4) -> Ast.TY_mach TY_i32
                | ("i64", DW_ATE_signed, 8) -> Ast.TY_mach TY_i64
                | ("char", DW_ATE_unsigned_char, 4) -> Ast.TY_char
                | ("int", DW_ATE_signed, sz)
                    when sz = word_sz_int -> Ast.TY_int
                | ("uint", DW_ATE_unsigned, sz)
                    when sz = word_sz_int -> Ast.TY_uint
                | _ -> bug () "unexpected type of DW_TAG_base_type"
            end

        | DW_TAG_structure_type ->
            begin
              let is_num_idx s =
                let len = String.length s in
                  if len >= 2 && s.[0] = '_'
                  then
                    let ok = ref true in
                      String.iter
                        (fun c -> ok := (!ok) && '0' <= c && c <= '9')
                        (String.sub s 1 (len-1));
                      !ok
                  else
                    false
              in
              let members = arr_map_partial
                die.die_children
                begin
                  fun child ->
                    if child.die_tag = DW_TAG_member
                    then Some child
                    else None
                end
              in
                assert ((Array.length members) > 0);
                if is_num_idx (get_name members.(0))
                then
                  let tys = Array.map get_referenced_ty members in
                    Ast.TY_tup tys
                else
                  let entries =
                    Array.map
                      (fun member_die -> ((get_name member_die),
                                          (get_referenced_ty member_die)))
                      members
                  in
                    Ast.TY_rec entries
            end

        | DW_TAG_interface_type ->
            let eff = get_effect die in
            let fns = Hashtbl.create 0 in
              Array.iter
                begin
                  fun child ->
                    if child.die_tag = DW_TAG_subroutine_type
                    then
                      Hashtbl.add fns (get_name child) (get_ty_fn child)
                end
                die.die_children;
              Ast.TY_obj (eff,fns)

        | DW_TAG_subroutine_type ->
            Ast.TY_fn (get_ty_fn die)

        | _ ->
            bug () "unexpected tag in get_ty: %s"
              (dw_tag_to_string die.die_tag)

  and get_slot die : Ast.slot =
    match die.die_tag with
        DW_TAG_reference_type ->
          let ty = get_referenced_ty die in
            { Ast.slot_mode = Ast.MODE_alias;
              Ast.slot_ty = Some ty }
      | _ ->
          let ty = get_ty die in
            { Ast.slot_mode = Ast.MODE_interior;
              Ast.slot_ty = Some ty }

  and get_referenced_ty die =
    match get_attr die DW_AT_type with
        (DW_FORM_ref_addr, DATA_num n) -> get_ty (get_die n)
      | _ -> bug () "unexpected form of DW_AT_type in get_referenced_ty"

  and get_referenced_slot die =
    match get_attr die DW_AT_type with
        (DW_FORM_ref_addr, DATA_num n) -> get_slot (get_die n)
      | _ -> bug () "unexpected form of DW_AT_type in get_referenced_slot"

  and get_ty_fn die =
    let out = get_referenced_slot die in
    let ins =
      arr_map_partial
        die.die_children
        begin
          fun child ->
            if child.die_tag = DW_TAG_formal_parameter
            then Some (get_referenced_slot child)
            else None
        end
    in
    let effect = get_effect die in
    let iter = get_flag die DW_AT_rust_iterator in
    let tsig =
      { Ast.sig_input_slots = ins;
        Ast.sig_input_constrs = [| |];
        Ast.sig_output_slot = out; }
    in
    let taux =
      { Ast.fn_is_iter = iter;
        Ast.fn_effect = effect }
    in
      (tsig, taux)
  in

  let wrap n =
    { id = next_node_id ();
      node = n }
  in

  let decl p i =
    wrap { Ast.decl_params = p;
           Ast.decl_item = i; }
  in

  let get_formals die =
    let islots = Queue.create () in
    let params = Queue.create () in
      Array.iter
        begin
          fun child ->
            match child.die_tag with
                DW_TAG_formal_parameter ->
                  if (is_rust_type child DW_RUST_type_param)
                  then Queue.push (wrap (get_type_param_decl child)) params
                  else Queue.push (get_referenced_slot child) islots
              | _ -> ()
        end
        die.die_children;
      (queue_to_arr params, queue_to_arr islots)
  in

  let extract_children mis die =
    Array.iter
      (fun child ->
         extract_mod_items nref oref abi mis (child.die_off,dies))
      die.die_children
  in

  let get_mod_items die =
    let len = Array.length die.die_children in
    let mis = Hashtbl.create len in
      extract_children mis die;
      mis
  in

  let form_header_slots slots =
    Array.mapi
      (fun i slot -> (wrap slot, "_" ^ (string_of_int i)))
      slots
  in

  let die = Hashtbl.find dies i in
    match die.die_tag with
        DW_TAG_typedef ->
          let ident = get_name die in
          let ty = get_referenced_ty die in
          let tyi = Ast.MOD_ITEM_type ty in
          let (params, islots) = get_formals die in
            assert ((Array.length islots) = 0);
            htab_put mis ident (decl params tyi)

      | DW_TAG_compile_unit ->
          extract_children mis die

      | DW_TAG_module ->
          let ident = get_name die in
          let sub_mis = get_mod_items die in
          let exports = Hashtbl.create 0 in
          let _ = Hashtbl.add exports Ast.EXPORT_all_decls () in
          let view = { Ast.view_imports = Hashtbl.create 0;
                       Ast.view_exports = exports }
          in
          let mi = Ast.MOD_ITEM_mod (view, sub_mis) in
            htab_put mis ident (decl [||] mi)

      | DW_TAG_subprogram ->
          (* FIXME (issue #74): finish this. *)
          let ident = get_name die in
          let oslot = get_referenced_slot die in
          let effect = get_effect die in
          let (params, islots) = get_formals die in
          let taux = { Ast.fn_effect = effect;
                       Ast.fn_is_iter = false }
          in
          let tfn = { Ast.fn_input_slots = form_header_slots islots;
                       Ast.fn_input_constrs = [| |];
                       Ast.fn_output_slot = wrap oslot;
                       Ast.fn_aux = taux;
                       Ast.fn_body = (wrap [||]); }
          in
          let fn = Ast.MOD_ITEM_fn tfn in
            htab_put mis ident (decl params fn)

      | _ -> ()
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

