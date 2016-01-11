/* Declarations and definitions of codes relating to the DWARF2 and
   DWARF3 symbolic debugging information formats.
   Copyright (C) 1992, 1993, 1995, 1996, 1997, 1999, 2000, 2001, 2002,
   2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012
   Free Software Foundation, Inc.

   Written by Gary Funck (gary@intrepid.com) The Ada Joint Program
   Office (AJPO), Florida State University and Silicon Graphics Inc.
   provided support for this effort -- June 21, 1995.

   Derived from the DWARF 1 implementation written by Ron Guilmette
   (rfg@netcom.com), November 1990.

   This file is part of GCC.

   GCC is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 3, or (at your option) any later
   version.

   GCC is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   Under Section 7 of GPL version 3, you are granted additional
   permissions described in the GCC Runtime Library Exception, version
   3.1, as published by the Free Software Foundation.

   You should have received a copy of the GNU General Public License and
   a copy of the GCC Runtime Library Exception along with this program;
   see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
   <http://www.gnu.org/licenses/>.  */

/* This file is derived from the DWARF specification (a public document)
   Revision 2.0.0 (July 27, 1993) developed by the UNIX International
   Programming Languages Special Interest Group (UI/PLSIG) and distributed
   by UNIX International.  Copies of this specification are available from
   UNIX International, 20 Waterview Boulevard, Parsippany, NJ, 07054.

   This file also now contains definitions from the DWARF 3 specification
   published Dec 20, 2005, available from: http://dwarf.freestandards.org.  */

#ifndef _DWARF2_H
#define _DWARF2_H

#define DW_TAG(name, value) , name = value
#define DW_TAG_DUP(name, value) , name = value
#define DW_FORM(name, value) , name = value
#define DW_AT(name, value) , name = value
#define DW_AT_DUP(name, value) , name = value
#define DW_OP(name, value) , name = value
#define DW_OP_DUP(name, value) , name = value
#define DW_ATE(name, value) , name = value
#define DW_ATE_DUP(name, value) , name = value
#define DW_CFA(name, value) , name = value

#define DW_FIRST_TAG(name, value) enum dwarf_tag { \
  name = value
#define DW_END_TAG };
#define DW_FIRST_FORM(name, value) enum dwarf_form { \
  name = value
#define DW_END_FORM };
#define DW_FIRST_AT(name, value) enum dwarf_attribute { \
  name = value
#define DW_END_AT };
#define DW_FIRST_OP(name, value) enum dwarf_location_atom { \
  name = value
#define DW_END_OP };
#define DW_FIRST_ATE(name, value) enum dwarf_type { \
  name = value
#define DW_END_ATE };
#define DW_FIRST_CFA(name, value) enum dwarf_call_frame_info { \
  name = value
#define DW_END_CFA };

#include "dwarf2.def"

#undef DW_FIRST_TAG
#undef DW_END_TAG
#undef DW_FIRST_FORM
#undef DW_END_FORM
#undef DW_FIRST_AT
#undef DW_END_AT
#undef DW_FIRST_OP
#undef DW_END_OP
#undef DW_FIRST_ATE
#undef DW_END_ATE
#undef DW_FIRST_CFA
#undef DW_END_CFA

#undef DW_TAG
#undef DW_TAG_DUP
#undef DW_FORM
#undef DW_AT
#undef DW_AT_DUP
#undef DW_OP
#undef DW_OP_DUP
#undef DW_ATE
#undef DW_ATE_DUP
#undef DW_CFA

/* Flag that tells whether entry has a child or not.  */
#define DW_children_no   0
#define	DW_children_yes  1

#define DW_AT_stride_size   DW_AT_bit_stride  /* Note: The use of DW_AT_stride_size is deprecated.  */
#define DW_AT_stride   DW_AT_byte_stride  /* Note: The use of DW_AT_stride is deprecated.  */

/* Decimal sign encodings.  */
enum dwarf_decimal_sign_encoding
  {
    /* DWARF 3.  */
    DW_DS_unsigned = 0x01,
    DW_DS_leading_overpunch = 0x02,
    DW_DS_trailing_overpunch = 0x03,
    DW_DS_leading_separate = 0x04,
    DW_DS_trailing_separate = 0x05
  };

/* Endianity encodings.  */
enum dwarf_endianity_encoding
  {
    /* DWARF 3.  */
    DW_END_default = 0x00,
    DW_END_big = 0x01,
    DW_END_little = 0x02,

    DW_END_lo_user = 0x40,
    DW_END_hi_user = 0xff
  };

/* Array ordering names and codes.  */
enum dwarf_array_dim_ordering
  {
    DW_ORD_row_major = 0,
    DW_ORD_col_major = 1
  };

/* Access attribute.  */
enum dwarf_access_attribute
  {
    DW_ACCESS_public = 1,
    DW_ACCESS_protected = 2,
    DW_ACCESS_private = 3
  };

/* Visibility.  */
enum dwarf_visibility_attribute
  {
    DW_VIS_local = 1,
    DW_VIS_exported = 2,
    DW_VIS_qualified = 3
  };

/* Virtuality.  */
enum dwarf_virtuality_attribute
  {
    DW_VIRTUALITY_none = 0,
    DW_VIRTUALITY_virtual = 1,
    DW_VIRTUALITY_pure_virtual = 2
  };

/* Case sensitivity.  */
enum dwarf_id_case
  {
    DW_ID_case_sensitive = 0,
    DW_ID_up_case = 1,
    DW_ID_down_case = 2,
    DW_ID_case_insensitive = 3
  };

/* Calling convention.  */
enum dwarf_calling_convention
  {
    DW_CC_normal = 0x1,
    DW_CC_program = 0x2,
    DW_CC_nocall = 0x3,

    DW_CC_lo_user = 0x40,
    DW_CC_hi_user = 0xff,

    DW_CC_GNU_renesas_sh = 0x40,
    DW_CC_GNU_borland_fastcall_i386 = 0x41,

    /* This DW_CC_ value is not currently generated by any toolchain.  It is
       used internally to GDB to indicate OpenCL C functions that have been
       compiled with the IBM XL C for OpenCL compiler and use a non-platform
       calling convention for passing OpenCL C vector types.  This value may
       be changed freely as long as it does not conflict with any other DW_CC_
       value defined here.  */
    DW_CC_GDB_IBM_OpenCL = 0xff
  };

/* Inline attribute.  */
enum dwarf_inline_attribute
  {
    DW_INL_not_inlined = 0,
    DW_INL_inlined = 1,
    DW_INL_declared_not_inlined = 2,
    DW_INL_declared_inlined = 3
  };

/* Discriminant lists.  */
enum dwarf_discrim_list
  {
    DW_DSC_label = 0,
    DW_DSC_range = 1
  };

/* Line number opcodes.  */
enum dwarf_line_number_ops
  {
    DW_LNS_extended_op = 0,
    DW_LNS_copy = 1,
    DW_LNS_advance_pc = 2,
    DW_LNS_advance_line = 3,
    DW_LNS_set_file = 4,
    DW_LNS_set_column = 5,
    DW_LNS_negate_stmt = 6,
    DW_LNS_set_basic_block = 7,
    DW_LNS_const_add_pc = 8,
    DW_LNS_fixed_advance_pc = 9,
    /* DWARF 3.  */
    DW_LNS_set_prologue_end = 10,
    DW_LNS_set_epilogue_begin = 11,
    DW_LNS_set_isa = 12
  };

/* Line number extended opcodes.  */
enum dwarf_line_number_x_ops
  {
    DW_LNE_end_sequence = 1,
    DW_LNE_set_address = 2,
    DW_LNE_define_file = 3,
    DW_LNE_set_discriminator = 4,
    /* HP extensions.  */
    DW_LNE_HP_negate_is_UV_update      = 0x11,
    DW_LNE_HP_push_context             = 0x12,
    DW_LNE_HP_pop_context              = 0x13,
    DW_LNE_HP_set_file_line_column     = 0x14,
    DW_LNE_HP_set_routine_name         = 0x15,
    DW_LNE_HP_set_sequence             = 0x16,
    DW_LNE_HP_negate_post_semantics    = 0x17,
    DW_LNE_HP_negate_function_exit     = 0x18,
    DW_LNE_HP_negate_front_end_logical = 0x19,
    DW_LNE_HP_define_proc              = 0x20,
    DW_LNE_HP_source_file_correlation  = 0x80,

    DW_LNE_lo_user = 0x80,
    DW_LNE_hi_user = 0xff
  };

/* Sub-opcodes for DW_LNE_HP_source_file_correlation.  */
enum dwarf_line_number_hp_sfc_ops
  {
    DW_LNE_HP_SFC_formfeed = 1,
    DW_LNE_HP_SFC_set_listing_line = 2,
    DW_LNE_HP_SFC_associate = 3
  };

/* Type codes for location list entries.
   Extension for Fission.  See http://gcc.gnu.org/wiki/DebugFission.  */

enum dwarf_location_list_entry_type
  {
    DW_LLE_GNU_end_of_list_entry = 0,
    DW_LLE_GNU_base_address_selection_entry = 1,
    DW_LLE_GNU_start_end_entry = 2,
    DW_LLE_GNU_start_length_entry = 3
  };

#define DW_CIE_ID	  0xffffffff
#define DW64_CIE_ID	  0xffffffffffffffffULL
#define DW_CIE_VERSION	  1

#define DW_CFA_extended   0

#define DW_CHILDREN_no		     0x00
#define DW_CHILDREN_yes		     0x01

#define DW_ADDR_none		0

/* Source language names and codes.  */
enum dwarf_source_language
  {
    DW_LANG_C89 = 0x0001,
    DW_LANG_C = 0x0002,
    DW_LANG_Ada83 = 0x0003,
    DW_LANG_C_plus_plus = 0x0004,
    DW_LANG_Cobol74 = 0x0005,
    DW_LANG_Cobol85 = 0x0006,
    DW_LANG_Fortran77 = 0x0007,
    DW_LANG_Fortran90 = 0x0008,
    DW_LANG_Pascal83 = 0x0009,
    DW_LANG_Modula2 = 0x000a,
    /* DWARF 3.  */
    DW_LANG_Java = 0x000b,
    DW_LANG_C99 = 0x000c,
    DW_LANG_Ada95 = 0x000d,
    DW_LANG_Fortran95 = 0x000e,
    DW_LANG_PLI = 0x000f,
    DW_LANG_ObjC = 0x0010,
    DW_LANG_ObjC_plus_plus = 0x0011,
    DW_LANG_UPC = 0x0012,
    DW_LANG_D = 0x0013,
    /* DWARF 4.  */
    DW_LANG_Python = 0x0014,
    /* DWARF 5.  */
    DW_LANG_Go = 0x0016,

    DW_LANG_lo_user = 0x8000,	/* Implementation-defined range start.  */
    DW_LANG_hi_user = 0xffff,	/* Implementation-defined range start.  */

    /* MIPS.  */
    DW_LANG_Mips_Assembler = 0x8001,
    /* UPC.  */
    DW_LANG_Upc = 0x8765,
    /* HP extensions.  */
    DW_LANG_HP_Bliss     = 0x8003,
    DW_LANG_HP_Basic91   = 0x8004,
    DW_LANG_HP_Pascal91  = 0x8005,
    DW_LANG_HP_IMacro    = 0x8006,
    DW_LANG_HP_Assembler = 0x8007
  };

/* Names and codes for macro information.  */
enum dwarf_macinfo_record_type
  {
    DW_MACINFO_define = 1,
    DW_MACINFO_undef = 2,
    DW_MACINFO_start_file = 3,
    DW_MACINFO_end_file = 4,
    DW_MACINFO_vendor_ext = 255
  };

/* Names and codes for new style macro information.  */
enum dwarf_macro_record_type
  {
    DW_MACRO_GNU_define = 1,
    DW_MACRO_GNU_undef = 2,
    DW_MACRO_GNU_start_file = 3,
    DW_MACRO_GNU_end_file = 4,
    DW_MACRO_GNU_define_indirect = 5,
    DW_MACRO_GNU_undef_indirect = 6,
    DW_MACRO_GNU_transparent_include = 7,
    /* Extensions for DWZ multifile.
       See http://www.dwarfstd.org/ShowIssue.php?issue=120604.1&type=open .  */
    DW_MACRO_GNU_define_indirect_alt = 8,
    DW_MACRO_GNU_undef_indirect_alt = 9,
    DW_MACRO_GNU_transparent_include_alt = 10,
    DW_MACRO_GNU_lo_user = 0xe0,
    DW_MACRO_GNU_hi_user = 0xff
  };

/* @@@ For use with GNU frame unwind information.  */

#define DW_EH_PE_absptr		0x00
#define DW_EH_PE_omit		0xff

#define DW_EH_PE_uleb128	0x01
#define DW_EH_PE_udata2		0x02
#define DW_EH_PE_udata4		0x03
#define DW_EH_PE_udata8		0x04
#define DW_EH_PE_sleb128	0x09
#define DW_EH_PE_sdata2		0x0A
#define DW_EH_PE_sdata4		0x0B
#define DW_EH_PE_sdata8		0x0C
#define DW_EH_PE_signed		0x08

#define DW_EH_PE_pcrel		0x10
#define DW_EH_PE_textrel	0x20
#define DW_EH_PE_datarel	0x30
#define DW_EH_PE_funcrel	0x40
#define DW_EH_PE_aligned	0x50

#define DW_EH_PE_indirect	0x80

/* Codes for the debug sections in a dwarf package (.dwp) file.
   Extensions for Fission.  See http://gcc.gnu.org/wiki/DebugFissionDWP.  */
enum dwarf_sect
  {
    DW_SECT_INFO = 1,
    DW_SECT_TYPES = 2,
    DW_SECT_ABBREV = 3,
    DW_SECT_LINE = 4,
    DW_SECT_LOC = 5,
    DW_SECT_STR_OFFSETS = 6,
    DW_SECT_MACINFO = 7,
    DW_SECT_MACRO = 8,
    DW_SECT_MAX = 8
  };

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Return the name of a DW_TAG_ constant, or NULL if the value is not
   recognized.  */
extern const char *get_DW_TAG_name (unsigned int tag);

/* Return the name of a DW_AT_ constant, or NULL if the value is not
   recognized.  */
extern const char *get_DW_AT_name (unsigned int attr);

/* Return the name of a DW_FORM_ constant, or NULL if the value is not
   recognized.  */
extern const char *get_DW_FORM_name (unsigned int form);

/* Return the name of a DW_OP_ constant, or NULL if the value is not
   recognized.  */
extern const char *get_DW_OP_name (unsigned int op);

/* Return the name of a DW_ATE_ constant, or NULL if the value is not
   recognized.  */
extern const char *get_DW_ATE_name (unsigned int enc);

/* Return the name of a DW_CFA_ constant, or NULL if the value is not
   recognized.  */
extern const char *get_DW_CFA_name (unsigned int opc);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _DWARF2_H */
