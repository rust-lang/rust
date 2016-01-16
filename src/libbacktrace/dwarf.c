/* dwarf.c -- Get file/line information from DWARF for backtraces.
   Copyright (C) 2012-2016 Free Software Foundation, Inc.
   Written by Ian Lance Taylor, Google.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    (1) Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer. 

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.  
    
    (3) The name of the author may not be used to
    endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.  */

#include "config.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "dwarf2.h"
#include "filenames.h"

#include "backtrace.h"
#include "internal.h"

#if !defined(HAVE_DECL_STRNLEN) || !HAVE_DECL_STRNLEN

/* If strnlen is not declared, provide our own version.  */

static size_t
xstrnlen (const char *s, size_t maxlen)
{
  size_t i;

  for (i = 0; i < maxlen; ++i)
    if (s[i] == '\0')
      break;
  return i;
}

#define strnlen xstrnlen

#endif

/* A buffer to read DWARF info.  */

struct dwarf_buf
{
  /* Buffer name for error messages.  */
  const char *name;
  /* Start of the buffer.  */
  const unsigned char *start;
  /* Next byte to read.  */
  const unsigned char *buf;
  /* The number of bytes remaining.  */
  size_t left;
  /* Whether the data is big-endian.  */
  int is_bigendian;
  /* Error callback routine.  */
  backtrace_error_callback error_callback;
  /* Data for error_callback.  */
  void *data;
  /* Non-zero if we've reported an underflow error.  */
  int reported_underflow;
};

/* A single attribute in a DWARF abbreviation.  */

struct attr
{
  /* The attribute name.  */
  enum dwarf_attribute name;
  /* The attribute form.  */
  enum dwarf_form form;
};

/* A single DWARF abbreviation.  */

struct abbrev
{
  /* The abbrev code--the number used to refer to the abbrev.  */
  uint64_t code;
  /* The entry tag.  */
  enum dwarf_tag tag;
  /* Non-zero if this abbrev has child entries.  */
  int has_children;
  /* The number of attributes.  */
  size_t num_attrs;
  /* The attributes.  */
  struct attr *attrs;
};

/* The DWARF abbreviations for a compilation unit.  This structure
   only exists while reading the compilation unit.  Most DWARF readers
   seem to a hash table to map abbrev ID's to abbrev entries.
   However, we primarily care about GCC, and GCC simply issues ID's in
   numerical order starting at 1.  So we simply keep a sorted vector,
   and try to just look up the code.  */

struct abbrevs
{
  /* The number of abbrevs in the vector.  */
  size_t num_abbrevs;
  /* The abbrevs, sorted by the code field.  */
  struct abbrev *abbrevs;
};

/* The different kinds of attribute values.  */

enum attr_val_encoding
{
  /* An address.  */
  ATTR_VAL_ADDRESS,
  /* A unsigned integer.  */
  ATTR_VAL_UINT,
  /* A sigd integer.  */
  ATTR_VAL_SINT,
  /* A string.  */
  ATTR_VAL_STRING,
  /* An offset to other data in the containing unit.  */
  ATTR_VAL_REF_UNIT,
  /* An offset to other data within the .dwarf_info section.  */
  ATTR_VAL_REF_INFO,
  /* An offset to data in some other section.  */
  ATTR_VAL_REF_SECTION,
  /* A type signature.  */
  ATTR_VAL_REF_TYPE,
  /* A block of data (not represented).  */
  ATTR_VAL_BLOCK,
  /* An expression (not represented).  */
  ATTR_VAL_EXPR,
};

/* An attribute value.  */

struct attr_val
{
  /* How the value is stored in the field u.  */
  enum attr_val_encoding encoding;
  union
  {
    /* ATTR_VAL_ADDRESS, ATTR_VAL_UINT, ATTR_VAL_REF*.  */
    uint64_t uint;
    /* ATTR_VAL_SINT.  */
    int64_t sint;
    /* ATTR_VAL_STRING.  */
    const char *string;
    /* ATTR_VAL_BLOCK not stored.  */
  } u;
};

/* The line number program header.  */

struct line_header
{
  /* The version of the line number information.  */
  int version;
  /* The minimum instruction length.  */
  unsigned int min_insn_len;
  /* The maximum number of ops per instruction.  */
  unsigned int max_ops_per_insn;
  /* The line base for special opcodes.  */
  int line_base;
  /* The line range for special opcodes.  */
  unsigned int line_range;
  /* The opcode base--the first special opcode.  */
  unsigned int opcode_base;
  /* Opcode lengths, indexed by opcode - 1.  */
  const unsigned char *opcode_lengths;
  /* The number of directory entries.  */
  size_t dirs_count;
  /* The directory entries.  */
  const char **dirs;
  /* The number of filenames.  */
  size_t filenames_count;
  /* The filenames.  */
  const char **filenames;
};

/* Map a single PC value to a file/line.  We will keep a vector of
   these sorted by PC value.  Each file/line will be correct from the
   PC up to the PC of the next entry if there is one.  We allocate one
   extra entry at the end so that we can use bsearch.  */

struct line
{
  /* PC.  */
  uintptr_t pc;
  /* File name.  Many entries in the array are expected to point to
     the same file name.  */
  const char *filename;
  /* Line number.  */
  int lineno;
  /* Index of the object in the original array read from the DWARF
     section, before it has been sorted.  The index makes it possible
     to use Quicksort and maintain stability.  */
  int idx;
};

/* A growable vector of line number information.  This is used while
   reading the line numbers.  */

struct line_vector
{
  /* Memory.  This is an array of struct line.  */
  struct backtrace_vector vec;
  /* Number of valid mappings.  */
  size_t count;
};

/* A function described in the debug info.  */

struct function
{
  /* The name of the function.  */
  const char *name;
  /* If this is an inlined function, the filename of the call
     site.  */
  const char *caller_filename;
  /* If this is an inlined function, the line number of the call
     site.  */
  int caller_lineno;
  /* Map PC ranges to inlined functions.  */
  struct function_addrs *function_addrs;
  size_t function_addrs_count;
};

/* An address range for a function.  This maps a PC value to a
   specific function.  */

struct function_addrs
{
  /* Range is LOW <= PC < HIGH.  */
  uint64_t low;
  uint64_t high;
  /* Function for this address range.  */
  struct function *function;
};

/* A growable vector of function address ranges.  */

struct function_vector
{
  /* Memory.  This is an array of struct function_addrs.  */
  struct backtrace_vector vec;
  /* Number of address ranges present.  */
  size_t count;
};

/* A DWARF compilation unit.  This only holds the information we need
   to map a PC to a file and line.  */

struct unit
{
  /* The first entry for this compilation unit.  */
  const unsigned char *unit_data;
  /* The length of the data for this compilation unit.  */
  size_t unit_data_len;
  /* The offset of UNIT_DATA from the start of the information for
     this compilation unit.  */
  size_t unit_data_offset;
  /* DWARF version.  */
  int version;
  /* Whether unit is DWARF64.  */
  int is_dwarf64;
  /* Address size.  */
  int addrsize;
  /* Offset into line number information.  */
  off_t lineoff;
  /* Primary source file.  */
  const char *filename;
  /* Compilation command working directory.  */
  const char *comp_dir;
  /* Absolute file name, only set if needed.  */
  const char *abs_filename;
  /* The abbreviations for this unit.  */
  struct abbrevs abbrevs;

  /* The fields above this point are read in during initialization and
     may be accessed freely.  The fields below this point are read in
     as needed, and therefore require care, as different threads may
     try to initialize them simultaneously.  */

  /* PC to line number mapping.  This is NULL if the values have not
     been read.  This is (struct line *) -1 if there was an error
     reading the values.  */
  struct line *lines;
  /* Number of entries in lines.  */
  size_t lines_count;
  /* PC ranges to function.  */
  struct function_addrs *function_addrs;
  size_t function_addrs_count;
};

/* An address range for a compilation unit.  This maps a PC value to a
   specific compilation unit.  Note that we invert the representation
   in DWARF: instead of listing the units and attaching a list of
   ranges, we list the ranges and have each one point to the unit.
   This lets us do a binary search to find the unit.  */

struct unit_addrs
{
  /* Range is LOW <= PC < HIGH.  */
  uint64_t low;
  uint64_t high;
  /* Compilation unit for this address range.  */
  struct unit *u;
};

/* A growable vector of compilation unit address ranges.  */

struct unit_addrs_vector
{
  /* Memory.  This is an array of struct unit_addrs.  */
  struct backtrace_vector vec;
  /* Number of address ranges present.  */
  size_t count;
};

/* The information we need to map a PC to a file and line.  */

struct dwarf_data
{
  /* The data for the next file we know about.  */
  struct dwarf_data *next;
  /* The base address for this file.  */
  uintptr_t base_address;
  /* A sorted list of address ranges.  */
  struct unit_addrs *addrs;
  /* Number of address ranges in list.  */
  size_t addrs_count;
  /* The unparsed .debug_info section.  */
  const unsigned char *dwarf_info;
  size_t dwarf_info_size;
  /* The unparsed .debug_line section.  */
  const unsigned char *dwarf_line;
  size_t dwarf_line_size;
  /* The unparsed .debug_ranges section.  */
  const unsigned char *dwarf_ranges;
  size_t dwarf_ranges_size;
  /* The unparsed .debug_str section.  */
  const unsigned char *dwarf_str;
  size_t dwarf_str_size;
  /* Whether the data is big-endian or not.  */
  int is_bigendian;
  /* A vector used for function addresses.  We keep this here so that
     we can grow the vector as we read more functions.  */
  struct function_vector fvec;
};

/* Report an error for a DWARF buffer.  */

static void
dwarf_buf_error (struct dwarf_buf *buf, const char *msg)
{
  char b[200];

  snprintf (b, sizeof b, "%s in %s at %d",
	    msg, buf->name, (int) (buf->buf - buf->start));
  buf->error_callback (buf->data, b, 0);
}

/* Require at least COUNT bytes in BUF.  Return 1 if all is well, 0 on
   error.  */

static int
require (struct dwarf_buf *buf, size_t count)
{
  if (buf->left >= count)
    return 1;

  if (!buf->reported_underflow)
    {
      dwarf_buf_error (buf, "DWARF underflow");
      buf->reported_underflow = 1;
    }

  return 0;
}

/* Advance COUNT bytes in BUF.  Return 1 if all is well, 0 on
   error.  */

static int
advance (struct dwarf_buf *buf, size_t count)
{
  if (!require (buf, count))
    return 0;
  buf->buf += count;
  buf->left -= count;
  return 1;
}

/* Read one byte from BUF and advance 1 byte.  */

static unsigned char
read_byte (struct dwarf_buf *buf)
{
  const unsigned char *p = buf->buf;

  if (!advance (buf, 1))
    return 0;
  return p[0];
}

/* Read a signed char from BUF and advance 1 byte.  */

static signed char
read_sbyte (struct dwarf_buf *buf)
{
  const unsigned char *p = buf->buf;

  if (!advance (buf, 1))
    return 0;
  return (*p ^ 0x80) - 0x80;
}

/* Read a uint16 from BUF and advance 2 bytes.  */

static uint16_t
read_uint16 (struct dwarf_buf *buf)
{
  const unsigned char *p = buf->buf;

  if (!advance (buf, 2))
    return 0;
  if (buf->is_bigendian)
    return ((uint16_t) p[0] << 8) | (uint16_t) p[1];
  else
    return ((uint16_t) p[1] << 8) | (uint16_t) p[0];
}

/* Read a uint32 from BUF and advance 4 bytes.  */

static uint32_t
read_uint32 (struct dwarf_buf *buf)
{
  const unsigned char *p = buf->buf;

  if (!advance (buf, 4))
    return 0;
  if (buf->is_bigendian)
    return (((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16)
	    | ((uint32_t) p[2] << 8) | (uint32_t) p[3]);
  else
    return (((uint32_t) p[3] << 24) | ((uint32_t) p[2] << 16)
	    | ((uint32_t) p[1] << 8) | (uint32_t) p[0]);
}

/* Read a uint64 from BUF and advance 8 bytes.  */

static uint64_t
read_uint64 (struct dwarf_buf *buf)
{
  const unsigned char *p = buf->buf;

  if (!advance (buf, 8))
    return 0;
  if (buf->is_bigendian)
    return (((uint64_t) p[0] << 56) | ((uint64_t) p[1] << 48)
	    | ((uint64_t) p[2] << 40) | ((uint64_t) p[3] << 32)
	    | ((uint64_t) p[4] << 24) | ((uint64_t) p[5] << 16)
	    | ((uint64_t) p[6] << 8) | (uint64_t) p[7]);
  else
    return (((uint64_t) p[7] << 56) | ((uint64_t) p[6] << 48)
	    | ((uint64_t) p[5] << 40) | ((uint64_t) p[4] << 32)
	    | ((uint64_t) p[3] << 24) | ((uint64_t) p[2] << 16)
	    | ((uint64_t) p[1] << 8) | (uint64_t) p[0]);
}

/* Read an offset from BUF and advance the appropriate number of
   bytes.  */

static uint64_t
read_offset (struct dwarf_buf *buf, int is_dwarf64)
{
  if (is_dwarf64)
    return read_uint64 (buf);
  else
    return read_uint32 (buf);
}

/* Read an address from BUF and advance the appropriate number of
   bytes.  */

static uint64_t
read_address (struct dwarf_buf *buf, int addrsize)
{
  switch (addrsize)
    {
    case 1:
      return read_byte (buf);
    case 2:
      return read_uint16 (buf);
    case 4:
      return read_uint32 (buf);
    case 8:
      return read_uint64 (buf);
    default:
      dwarf_buf_error (buf, "unrecognized address size");
      return 0;
    }
}

/* Return whether a value is the highest possible address, given the
   address size.  */

static int
is_highest_address (uint64_t address, int addrsize)
{
  switch (addrsize)
    {
    case 1:
      return address == (unsigned char) -1;
    case 2:
      return address == (uint16_t) -1;
    case 4:
      return address == (uint32_t) -1;
    case 8:
      return address == (uint64_t) -1;
    default:
      return 0;
    }
}

/* Read an unsigned LEB128 number.  */

static uint64_t
read_uleb128 (struct dwarf_buf *buf)
{
  uint64_t ret;
  unsigned int shift;
  int overflow;
  unsigned char b;

  ret = 0;
  shift = 0;
  overflow = 0;
  do
    {
      const unsigned char *p;

      p = buf->buf;
      if (!advance (buf, 1))
	return 0;
      b = *p;
      if (shift < 64)
	ret |= ((uint64_t) (b & 0x7f)) << shift;
      else if (!overflow)
	{
	  dwarf_buf_error (buf, "LEB128 overflows uint64_t");
	  overflow = 1;
	}
      shift += 7;
    }
  while ((b & 0x80) != 0);

  return ret;
}

/* Read a signed LEB128 number.  */

static int64_t
read_sleb128 (struct dwarf_buf *buf)
{
  uint64_t val;
  unsigned int shift;
  int overflow;
  unsigned char b;

  val = 0;
  shift = 0;
  overflow = 0;
  do
    {
      const unsigned char *p;

      p = buf->buf;
      if (!advance (buf, 1))
	return 0;
      b = *p;
      if (shift < 64)
	val |= ((uint64_t) (b & 0x7f)) << shift;
      else if (!overflow)
	{
	  dwarf_buf_error (buf, "signed LEB128 overflows uint64_t");
	  overflow = 1;
	}
      shift += 7;
    }
  while ((b & 0x80) != 0);

  if ((b & 0x40) != 0 && shift < 64)
    val |= ((uint64_t) -1) << shift;

  return (int64_t) val;
}

/* Return the length of an LEB128 number.  */

static size_t
leb128_len (const unsigned char *p)
{
  size_t ret;

  ret = 1;
  while ((*p & 0x80) != 0)
    {
      ++p;
      ++ret;
    }
  return ret;
}

/* Free an abbreviations structure.  */

static void
free_abbrevs (struct backtrace_state *state, struct abbrevs *abbrevs,
	      backtrace_error_callback error_callback, void *data)
{
  size_t i;

  for (i = 0; i < abbrevs->num_abbrevs; ++i)
    backtrace_free (state, abbrevs->abbrevs[i].attrs,
		    abbrevs->abbrevs[i].num_attrs * sizeof (struct attr),
		    error_callback, data);
  backtrace_free (state, abbrevs->abbrevs,
		  abbrevs->num_abbrevs * sizeof (struct abbrev),
		  error_callback, data);
  abbrevs->num_abbrevs = 0;
  abbrevs->abbrevs = NULL;
}

/* Read an attribute value.  Returns 1 on success, 0 on failure.  If
   the value can be represented as a uint64_t, sets *VAL and sets
   *IS_VALID to 1.  We don't try to store the value of other attribute
   forms, because we don't care about them.  */

static int
read_attribute (enum dwarf_form form, struct dwarf_buf *buf,
		int is_dwarf64, int version, int addrsize,
		const unsigned char *dwarf_str, size_t dwarf_str_size,
		struct attr_val *val)
{
  /* Avoid warnings about val.u.FIELD may be used uninitialized if
     this function is inlined.  The warnings aren't valid but can
     occur because the different fields are set and used
     conditionally.  */
  memset (val, 0, sizeof *val);

  switch (form)
    {
    case DW_FORM_addr:
      val->encoding = ATTR_VAL_ADDRESS;
      val->u.uint = read_address (buf, addrsize);
      return 1;
    case DW_FORM_block2:
      val->encoding = ATTR_VAL_BLOCK;
      return advance (buf, read_uint16 (buf));
    case DW_FORM_block4:
      val->encoding = ATTR_VAL_BLOCK;
      return advance (buf, read_uint32 (buf));
    case DW_FORM_data2:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = read_uint16 (buf);
      return 1;
    case DW_FORM_data4:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = read_uint32 (buf);
      return 1;
    case DW_FORM_data8:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = read_uint64 (buf);
      return 1;
    case DW_FORM_string:
      val->encoding = ATTR_VAL_STRING;
      val->u.string = (const char *) buf->buf;
      return advance (buf, strnlen ((const char *) buf->buf, buf->left) + 1);
    case DW_FORM_block:
      val->encoding = ATTR_VAL_BLOCK;
      return advance (buf, read_uleb128 (buf));
    case DW_FORM_block1:
      val->encoding = ATTR_VAL_BLOCK;
      return advance (buf, read_byte (buf));
    case DW_FORM_data1:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = read_byte (buf);
      return 1;
    case DW_FORM_flag:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = read_byte (buf);
      return 1;
    case DW_FORM_sdata:
      val->encoding = ATTR_VAL_SINT;
      val->u.sint = read_sleb128 (buf);
      return 1;
    case DW_FORM_strp:
      {
	uint64_t offset;

	offset = read_offset (buf, is_dwarf64);
	if (offset >= dwarf_str_size)
	  {
	    dwarf_buf_error (buf, "DW_FORM_strp out of range");
	    return 0;
	  }
	val->encoding = ATTR_VAL_STRING;
	val->u.string = (const char *) dwarf_str + offset;
	return 1;
      }
    case DW_FORM_udata:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = read_uleb128 (buf);
      return 1;
    case DW_FORM_ref_addr:
      val->encoding = ATTR_VAL_REF_INFO;
      if (version == 2)
	val->u.uint = read_address (buf, addrsize);
      else
	val->u.uint = read_offset (buf, is_dwarf64);
      return 1;
    case DW_FORM_ref1:
      val->encoding = ATTR_VAL_REF_UNIT;
      val->u.uint = read_byte (buf);
      return 1;
    case DW_FORM_ref2:
      val->encoding = ATTR_VAL_REF_UNIT;
      val->u.uint = read_uint16 (buf);
      return 1;
    case DW_FORM_ref4:
      val->encoding = ATTR_VAL_REF_UNIT;
      val->u.uint = read_uint32 (buf);
      return 1;
    case DW_FORM_ref8:
      val->encoding = ATTR_VAL_REF_UNIT;
      val->u.uint = read_uint64 (buf);
      return 1;
    case DW_FORM_ref_udata:
      val->encoding = ATTR_VAL_REF_UNIT;
      val->u.uint = read_uleb128 (buf);
      return 1;
    case DW_FORM_indirect:
      {
	uint64_t form;

	form = read_uleb128 (buf);
	return read_attribute ((enum dwarf_form) form, buf, is_dwarf64,
			       version, addrsize, dwarf_str, dwarf_str_size,
			       val);
      }
    case DW_FORM_sec_offset:
      val->encoding = ATTR_VAL_REF_SECTION;
      val->u.uint = read_offset (buf, is_dwarf64);
      return 1;
    case DW_FORM_exprloc:
      val->encoding = ATTR_VAL_EXPR;
      return advance (buf, read_uleb128 (buf));
    case DW_FORM_flag_present:
      val->encoding = ATTR_VAL_UINT;
      val->u.uint = 1;
      return 1;
    case DW_FORM_ref_sig8:
      val->encoding = ATTR_VAL_REF_TYPE;
      val->u.uint = read_uint64 (buf);
      return 1;
    case DW_FORM_GNU_addr_index:
      val->encoding = ATTR_VAL_REF_SECTION;
      val->u.uint = read_uleb128 (buf);
      return 1;
    case DW_FORM_GNU_str_index:
      val->encoding = ATTR_VAL_REF_SECTION;
      val->u.uint = read_uleb128 (buf);
      return 1;
    case DW_FORM_GNU_ref_alt:
      val->encoding = ATTR_VAL_REF_SECTION;
      val->u.uint = read_offset (buf, is_dwarf64);
      return 1;
    case DW_FORM_GNU_strp_alt:
      val->encoding = ATTR_VAL_REF_SECTION;
      val->u.uint = read_offset (buf, is_dwarf64);
      return 1;
    default:
      dwarf_buf_error (buf, "unrecognized DWARF form");
      return 0;
    }
}

/* Compare function_addrs for qsort.  When ranges are nested, make the
   smallest one sort last.  */

static int
function_addrs_compare (const void *v1, const void *v2)
{
  const struct function_addrs *a1 = (const struct function_addrs *) v1;
  const struct function_addrs *a2 = (const struct function_addrs *) v2;

  if (a1->low < a2->low)
    return -1;
  if (a1->low > a2->low)
    return 1;
  if (a1->high < a2->high)
    return 1;
  if (a1->high > a2->high)
    return -1;
  return strcmp (a1->function->name, a2->function->name);
}

/* Compare a PC against a function_addrs for bsearch.  Note that if
   there are multiple ranges containing PC, which one will be returned
   is unpredictable.  We compensate for that in dwarf_fileline.  */

static int
function_addrs_search (const void *vkey, const void *ventry)
{
  const uintptr_t *key = (const uintptr_t *) vkey;
  const struct function_addrs *entry = (const struct function_addrs *) ventry;
  uintptr_t pc;

  pc = *key;
  if (pc < entry->low)
    return -1;
  else if (pc >= entry->high)
    return 1;
  else
    return 0;
}

/* Add a new compilation unit address range to a vector.  Returns 1 on
   success, 0 on failure.  */

static int
add_unit_addr (struct backtrace_state *state, uintptr_t base_address,
	       struct unit_addrs addrs,
	       backtrace_error_callback error_callback, void *data,
	       struct unit_addrs_vector *vec)
{
  struct unit_addrs *p;

  /* Add in the base address of the module here, so that we can look
     up the PC directly.  */
  addrs.low += base_address;
  addrs.high += base_address;

  /* Try to merge with the last entry.  */
  if (vec->count > 0)
    {
      p = (struct unit_addrs *) vec->vec.base + (vec->count - 1);
      if ((addrs.low == p->high || addrs.low == p->high + 1)
	  && addrs.u == p->u)
	{
	  if (addrs.high > p->high)
	    p->high = addrs.high;
	  return 1;
	}
    }

  p = ((struct unit_addrs *)
       backtrace_vector_grow (state, sizeof (struct unit_addrs),
			      error_callback, data, &vec->vec));
  if (p == NULL)
    return 0;

  *p = addrs;
  ++vec->count;
  return 1;
}

/* Free a unit address vector.  */

static void
free_unit_addrs_vector (struct backtrace_state *state,
			struct unit_addrs_vector *vec,
			backtrace_error_callback error_callback, void *data)
{
  struct unit_addrs *addrs;
  size_t i;

  addrs = (struct unit_addrs *) vec->vec.base;
  for (i = 0; i < vec->count; ++i)
    free_abbrevs (state, &addrs[i].u->abbrevs, error_callback, data);
}

/* Compare unit_addrs for qsort.  When ranges are nested, make the
   smallest one sort last.  */

static int
unit_addrs_compare (const void *v1, const void *v2)
{
  const struct unit_addrs *a1 = (const struct unit_addrs *) v1;
  const struct unit_addrs *a2 = (const struct unit_addrs *) v2;

  if (a1->low < a2->low)
    return -1;
  if (a1->low > a2->low)
    return 1;
  if (a1->high < a2->high)
    return 1;
  if (a1->high > a2->high)
    return -1;
  if (a1->u->lineoff < a2->u->lineoff)
    return -1;
  if (a1->u->lineoff > a2->u->lineoff)
    return 1;
  return 0;
}

/* Compare a PC against a unit_addrs for bsearch.  Note that if there
   are multiple ranges containing PC, which one will be returned is
   unpredictable.  We compensate for that in dwarf_fileline.  */

static int
unit_addrs_search (const void *vkey, const void *ventry)
{
  const uintptr_t *key = (const uintptr_t *) vkey;
  const struct unit_addrs *entry = (const struct unit_addrs *) ventry;
  uintptr_t pc;

  pc = *key;
  if (pc < entry->low)
    return -1;
  else if (pc >= entry->high)
    return 1;
  else
    return 0;
}

/* Sort the line vector by PC.  We want a stable sort here to maintain
   the order of lines for the same PC values.  Since the sequence is
   being sorted in place, their addresses cannot be relied on to
   maintain stability.  That is the purpose of the index member.  */

static int
line_compare (const void *v1, const void *v2)
{
  const struct line *ln1 = (const struct line *) v1;
  const struct line *ln2 = (const struct line *) v2;

  if (ln1->pc < ln2->pc)
    return -1;
  else if (ln1->pc > ln2->pc)
    return 1;
  else if (ln1->idx < ln2->idx)
    return -1;
  else if (ln1->idx > ln2->idx)
    return 1;
  else
    return 0;
}

/* Find a PC in a line vector.  We always allocate an extra entry at
   the end of the lines vector, so that this routine can safely look
   at the next entry.  Note that when there are multiple mappings for
   the same PC value, this will return the last one.  */

static int
line_search (const void *vkey, const void *ventry)
{
  const uintptr_t *key = (const uintptr_t *) vkey;
  const struct line *entry = (const struct line *) ventry;
  uintptr_t pc;

  pc = *key;
  if (pc < entry->pc)
    return -1;
  else if (pc >= (entry + 1)->pc)
    return 1;
  else
    return 0;
}

/* Sort the abbrevs by the abbrev code.  This function is passed to
   both qsort and bsearch.  */

static int
abbrev_compare (const void *v1, const void *v2)
{
  const struct abbrev *a1 = (const struct abbrev *) v1;
  const struct abbrev *a2 = (const struct abbrev *) v2;

  if (a1->code < a2->code)
    return -1;
  else if (a1->code > a2->code)
    return 1;
  else
    {
      /* This really shouldn't happen.  It means there are two
	 different abbrevs with the same code, and that means we don't
	 know which one lookup_abbrev should return.  */
      return 0;
    }
}

/* Read the abbreviation table for a compilation unit.  Returns 1 on
   success, 0 on failure.  */

static int
read_abbrevs (struct backtrace_state *state, uint64_t abbrev_offset,
	      const unsigned char *dwarf_abbrev, size_t dwarf_abbrev_size,
	      int is_bigendian, backtrace_error_callback error_callback,
	      void *data, struct abbrevs *abbrevs)
{
  struct dwarf_buf abbrev_buf;
  struct dwarf_buf count_buf;
  size_t num_abbrevs;

  abbrevs->num_abbrevs = 0;
  abbrevs->abbrevs = NULL;

  if (abbrev_offset >= dwarf_abbrev_size)
    {
      error_callback (data, "abbrev offset out of range", 0);
      return 0;
    }

  abbrev_buf.name = ".debug_abbrev";
  abbrev_buf.start = dwarf_abbrev;
  abbrev_buf.buf = dwarf_abbrev + abbrev_offset;
  abbrev_buf.left = dwarf_abbrev_size - abbrev_offset;
  abbrev_buf.is_bigendian = is_bigendian;
  abbrev_buf.error_callback = error_callback;
  abbrev_buf.data = data;
  abbrev_buf.reported_underflow = 0;

  /* Count the number of abbrevs in this list.  */

  count_buf = abbrev_buf;
  num_abbrevs = 0;
  while (read_uleb128 (&count_buf) != 0)
    {
      if (count_buf.reported_underflow)
	return 0;
      ++num_abbrevs;
      // Skip tag.
      read_uleb128 (&count_buf);
      // Skip has_children.
      read_byte (&count_buf);
      // Skip attributes.
      while (read_uleb128 (&count_buf) != 0)
	read_uleb128 (&count_buf);
      // Skip form of last attribute.
      read_uleb128 (&count_buf);
    }

  if (count_buf.reported_underflow)
    return 0;

  if (num_abbrevs == 0)
    return 1;

  abbrevs->num_abbrevs = num_abbrevs;
  abbrevs->abbrevs = ((struct abbrev *)
		      backtrace_alloc (state,
				       num_abbrevs * sizeof (struct abbrev),
				       error_callback, data));
  if (abbrevs->abbrevs == NULL)
    return 0;
  memset (abbrevs->abbrevs, 0, num_abbrevs * sizeof (struct abbrev));

  num_abbrevs = 0;
  while (1)
    {
      uint64_t code;
      struct abbrev a;
      size_t num_attrs;
      struct attr *attrs;

      if (abbrev_buf.reported_underflow)
	goto fail;

      code = read_uleb128 (&abbrev_buf);
      if (code == 0)
	break;

      a.code = code;
      a.tag = (enum dwarf_tag) read_uleb128 (&abbrev_buf);
      a.has_children = read_byte (&abbrev_buf);

      count_buf = abbrev_buf;
      num_attrs = 0;
      while (read_uleb128 (&count_buf) != 0)
	{
	  ++num_attrs;
	  read_uleb128 (&count_buf);
	}

      if (num_attrs == 0)
	{
	  attrs = NULL;
	  read_uleb128 (&abbrev_buf);
	  read_uleb128 (&abbrev_buf);
	}
      else
	{
	  attrs = ((struct attr *)
		   backtrace_alloc (state, num_attrs * sizeof *attrs,
				    error_callback, data));
	  if (attrs == NULL)
	    goto fail;
	  num_attrs = 0;
	  while (1)
	    {
	      uint64_t name;
	      uint64_t form;

	      name = read_uleb128 (&abbrev_buf);
	      form = read_uleb128 (&abbrev_buf);
	      if (name == 0)
		break;
	      attrs[num_attrs].name = (enum dwarf_attribute) name;
	      attrs[num_attrs].form = (enum dwarf_form) form;
	      ++num_attrs;
	    }
	}

      a.num_attrs = num_attrs;
      a.attrs = attrs;

      abbrevs->abbrevs[num_abbrevs] = a;
      ++num_abbrevs;
    }

  backtrace_qsort (abbrevs->abbrevs, abbrevs->num_abbrevs,
		   sizeof (struct abbrev), abbrev_compare);

  return 1;

 fail:
  free_abbrevs (state, abbrevs, error_callback, data);
  return 0;
}

/* Return the abbrev information for an abbrev code.  */

static const struct abbrev *
lookup_abbrev (struct abbrevs *abbrevs, uint64_t code,
	       backtrace_error_callback error_callback, void *data)
{
  struct abbrev key;
  void *p;

  /* With GCC, where abbrevs are simply numbered in order, we should
     be able to just look up the entry.  */
  if (code - 1 < abbrevs->num_abbrevs
      && abbrevs->abbrevs[code - 1].code == code)
    return &abbrevs->abbrevs[code - 1];

  /* Otherwise we have to search.  */
  memset (&key, 0, sizeof key);
  key.code = code;
  p = bsearch (&key, abbrevs->abbrevs, abbrevs->num_abbrevs,
	       sizeof (struct abbrev), abbrev_compare);
  if (p == NULL)
    {
      error_callback (data, "invalid abbreviation code", 0);
      return NULL;
    }
  return (const struct abbrev *) p;
}

/* Add non-contiguous address ranges for a compilation unit.  Returns
   1 on success, 0 on failure.  */

static int
add_unit_ranges (struct backtrace_state *state, uintptr_t base_address,
		 struct unit *u, uint64_t ranges, uint64_t base,
		 int is_bigendian, const unsigned char *dwarf_ranges,
		 size_t dwarf_ranges_size,
		 backtrace_error_callback error_callback, void *data,
		 struct unit_addrs_vector *addrs)
{
  struct dwarf_buf ranges_buf;

  if (ranges >= dwarf_ranges_size)
    {
      error_callback (data, "ranges offset out of range", 0);
      return 0;
    }

  ranges_buf.name = ".debug_ranges";
  ranges_buf.start = dwarf_ranges;
  ranges_buf.buf = dwarf_ranges + ranges;
  ranges_buf.left = dwarf_ranges_size - ranges;
  ranges_buf.is_bigendian = is_bigendian;
  ranges_buf.error_callback = error_callback;
  ranges_buf.data = data;
  ranges_buf.reported_underflow = 0;

  while (1)
    {
      uint64_t low;
      uint64_t high;

      if (ranges_buf.reported_underflow)
	return 0;

      low = read_address (&ranges_buf, u->addrsize);
      high = read_address (&ranges_buf, u->addrsize);

      if (low == 0 && high == 0)
	break;

      if (is_highest_address (low, u->addrsize))
	base = high;
      else
	{
	  struct unit_addrs a;

	  a.low = low + base;
	  a.high = high + base;
	  a.u = u;
	  if (!add_unit_addr (state, base_address, a, error_callback, data,
			      addrs))
	    return 0;
	}
    }

  if (ranges_buf.reported_underflow)
    return 0;

  return 1;
}

/* Find the address range covered by a compilation unit, reading from
   UNIT_BUF and adding values to U.  Returns 1 if all data could be
   read, 0 if there is some error.  */

static int
find_address_ranges (struct backtrace_state *state, uintptr_t base_address,
		     struct dwarf_buf *unit_buf, 
		     const unsigned char *dwarf_str, size_t dwarf_str_size,
		     const unsigned char *dwarf_ranges,
		     size_t dwarf_ranges_size,
		     int is_bigendian, backtrace_error_callback error_callback,
		     void *data, struct unit *u,
		     struct unit_addrs_vector *addrs)
{
  while (unit_buf->left > 0)
    {
      uint64_t code;
      const struct abbrev *abbrev;
      uint64_t lowpc;
      int have_lowpc;
      uint64_t highpc;
      int have_highpc;
      int highpc_is_relative;
      uint64_t ranges;
      int have_ranges;
      size_t i;

      code = read_uleb128 (unit_buf);
      if (code == 0)
	return 1;

      abbrev = lookup_abbrev (&u->abbrevs, code, error_callback, data);
      if (abbrev == NULL)
	return 0;

      lowpc = 0;
      have_lowpc = 0;
      highpc = 0;
      have_highpc = 0;
      highpc_is_relative = 0;
      ranges = 0;
      have_ranges = 0;
      for (i = 0; i < abbrev->num_attrs; ++i)
	{
	  struct attr_val val;

	  if (!read_attribute (abbrev->attrs[i].form, unit_buf,
			       u->is_dwarf64, u->version, u->addrsize,
			       dwarf_str, dwarf_str_size, &val))
	    return 0;

	  switch (abbrev->attrs[i].name)
	    {
	    case DW_AT_low_pc:
	      if (val.encoding == ATTR_VAL_ADDRESS)
		{
		  lowpc = val.u.uint;
		  have_lowpc = 1;
		}
	      break;

	    case DW_AT_high_pc:
	      if (val.encoding == ATTR_VAL_ADDRESS)
		{
		  highpc = val.u.uint;
		  have_highpc = 1;
		}
	      else if (val.encoding == ATTR_VAL_UINT)
		{
		  highpc = val.u.uint;
		  have_highpc = 1;
		  highpc_is_relative = 1;
		}
	      break;

	    case DW_AT_ranges:
	      if (val.encoding == ATTR_VAL_UINT
		  || val.encoding == ATTR_VAL_REF_SECTION)
		{
		  ranges = val.u.uint;
		  have_ranges = 1;
		}
	      break;

	    case DW_AT_stmt_list:
	      if (abbrev->tag == DW_TAG_compile_unit
		  && (val.encoding == ATTR_VAL_UINT
		      || val.encoding == ATTR_VAL_REF_SECTION))
		u->lineoff = val.u.uint;
	      break;

	    case DW_AT_name:
	      if (abbrev->tag == DW_TAG_compile_unit
		  && val.encoding == ATTR_VAL_STRING)
		u->filename = val.u.string;
	      break;

	    case DW_AT_comp_dir:
	      if (abbrev->tag == DW_TAG_compile_unit
		  && val.encoding == ATTR_VAL_STRING)
		u->comp_dir = val.u.string;
	      break;

	    default:
	      break;
	    }
	}

      if (abbrev->tag == DW_TAG_compile_unit
	  || abbrev->tag == DW_TAG_subprogram)
	{
	  if (have_ranges)
	    {
	      if (!add_unit_ranges (state, base_address, u, ranges, lowpc,
				    is_bigendian, dwarf_ranges,
				    dwarf_ranges_size, error_callback,
				    data, addrs))
		return 0;
	    }
	  else if (have_lowpc && have_highpc)
	    {
	      struct unit_addrs a;

	      if (highpc_is_relative)
		highpc += lowpc;
	      a.low = lowpc;
	      a.high = highpc;
	      a.u = u;

	      if (!add_unit_addr (state, base_address, a, error_callback, data,
				  addrs))
		return 0;
	    }

	  /* If we found the PC range in the DW_TAG_compile_unit, we
	     can stop now.  */
	  if (abbrev->tag == DW_TAG_compile_unit
	      && (have_ranges || (have_lowpc && have_highpc)))
	    return 1;
	}

      if (abbrev->has_children)
	{
	  if (!find_address_ranges (state, base_address, unit_buf,
				    dwarf_str, dwarf_str_size,
				    dwarf_ranges, dwarf_ranges_size,
				    is_bigendian, error_callback, data,
				    u, addrs))
	    return 0;
	}
    }

  return 1;
}

/* Build a mapping from address ranges to the compilation units where
   the line number information for that range can be found.  Returns 1
   on success, 0 on failure.  */

static int
build_address_map (struct backtrace_state *state, uintptr_t base_address,
		   const unsigned char *dwarf_info, size_t dwarf_info_size,
		   const unsigned char *dwarf_abbrev, size_t dwarf_abbrev_size,
		   const unsigned char *dwarf_ranges, size_t dwarf_ranges_size,
		   const unsigned char *dwarf_str, size_t dwarf_str_size,
		   int is_bigendian, backtrace_error_callback error_callback,
		   void *data, struct unit_addrs_vector *addrs)
{
  struct dwarf_buf info;
  struct abbrevs abbrevs;

  memset (&addrs->vec, 0, sizeof addrs->vec);
  addrs->count = 0;

  /* Read through the .debug_info section.  FIXME: Should we use the
     .debug_aranges section?  gdb and addr2line don't use it, but I'm
     not sure why.  */

  info.name = ".debug_info";
  info.start = dwarf_info;
  info.buf = dwarf_info;
  info.left = dwarf_info_size;
  info.is_bigendian = is_bigendian;
  info.error_callback = error_callback;
  info.data = data;
  info.reported_underflow = 0;

  memset (&abbrevs, 0, sizeof abbrevs);
  while (info.left > 0)
    {
      const unsigned char *unit_data_start;
      uint64_t len;
      int is_dwarf64;
      struct dwarf_buf unit_buf;
      int version;
      uint64_t abbrev_offset;
      int addrsize;
      struct unit *u;

      if (info.reported_underflow)
	goto fail;

      unit_data_start = info.buf;

      is_dwarf64 = 0;
      len = read_uint32 (&info);
      if (len == 0xffffffff)
	{
	  len = read_uint64 (&info);
	  is_dwarf64 = 1;
	}

      unit_buf = info;
      unit_buf.left = len;

      if (!advance (&info, len))
	goto fail;

      version = read_uint16 (&unit_buf);
      if (version < 2 || version > 4)
	{
	  dwarf_buf_error (&unit_buf, "unrecognized DWARF version");
	  goto fail;
	}

      abbrev_offset = read_offset (&unit_buf, is_dwarf64);
      if (!read_abbrevs (state, abbrev_offset, dwarf_abbrev, dwarf_abbrev_size,
			 is_bigendian, error_callback, data, &abbrevs))
	goto fail;

      addrsize = read_byte (&unit_buf);

      u = ((struct unit *)
	   backtrace_alloc (state, sizeof *u, error_callback, data));
      if (u == NULL)
	goto fail;
      u->unit_data = unit_buf.buf;
      u->unit_data_len = unit_buf.left;
      u->unit_data_offset = unit_buf.buf - unit_data_start;
      u->version = version;
      u->is_dwarf64 = is_dwarf64;
      u->addrsize = addrsize;
      u->filename = NULL;
      u->comp_dir = NULL;
      u->abs_filename = NULL;
      u->lineoff = 0;
      u->abbrevs = abbrevs;
      memset (&abbrevs, 0, sizeof abbrevs);

      /* The actual line number mappings will be read as needed.  */
      u->lines = NULL;
      u->lines_count = 0;
      u->function_addrs = NULL;
      u->function_addrs_count = 0;

      if (!find_address_ranges (state, base_address, &unit_buf,
				dwarf_str, dwarf_str_size,
				dwarf_ranges, dwarf_ranges_size,
				is_bigendian, error_callback, data,
				u, addrs))
	{
	  free_abbrevs (state, &u->abbrevs, error_callback, data);
	  backtrace_free (state, u, sizeof *u, error_callback, data);
	  goto fail;
	}

      if (unit_buf.reported_underflow)
	{
	  free_abbrevs (state, &u->abbrevs, error_callback, data);
	  backtrace_free (state, u, sizeof *u, error_callback, data);
	  goto fail;
	}
    }
  if (info.reported_underflow)
    goto fail;

  return 1;

 fail:
  free_abbrevs (state, &abbrevs, error_callback, data);
  free_unit_addrs_vector (state, addrs, error_callback, data);
  return 0;
}

/* Add a new mapping to the vector of line mappings that we are
   building.  Returns 1 on success, 0 on failure.  */

static int
add_line (struct backtrace_state *state, struct dwarf_data *ddata,
	  uintptr_t pc, const char *filename, int lineno,
	  backtrace_error_callback error_callback, void *data,
	  struct line_vector *vec)
{
  struct line *ln;

  /* If we are adding the same mapping, ignore it.  This can happen
     when using discriminators.  */
  if (vec->count > 0)
    {
      ln = (struct line *) vec->vec.base + (vec->count - 1);
      if (pc == ln->pc && filename == ln->filename && lineno == ln->lineno)
	return 1;
    }

  ln = ((struct line *)
	backtrace_vector_grow (state, sizeof (struct line), error_callback,
			       data, &vec->vec));
  if (ln == NULL)
    return 0;

  /* Add in the base address here, so that we can look up the PC
     directly.  */
  ln->pc = pc + ddata->base_address;

  ln->filename = filename;
  ln->lineno = lineno;
  ln->idx = vec->count;

  ++vec->count;

  return 1;
}

/* Free the line header information.  If FREE_FILENAMES is true we
   free the file names themselves, otherwise we leave them, as there
   may be line structures pointing to them.  */

static void
free_line_header (struct backtrace_state *state, struct line_header *hdr,
		  backtrace_error_callback error_callback, void *data)
{
  backtrace_free (state, hdr->dirs, hdr->dirs_count * sizeof (const char *),
		  error_callback, data);
  backtrace_free (state, hdr->filenames,
		  hdr->filenames_count * sizeof (char *),
		  error_callback, data);
}

/* Read the line header.  Return 1 on success, 0 on failure.  */

static int
read_line_header (struct backtrace_state *state, struct unit *u,
		  int is_dwarf64, struct dwarf_buf *line_buf,
		  struct line_header *hdr)
{
  uint64_t hdrlen;
  struct dwarf_buf hdr_buf;
  const unsigned char *p;
  const unsigned char *pend;
  size_t i;

  hdr->version = read_uint16 (line_buf);
  if (hdr->version < 2 || hdr->version > 4)
    {
      dwarf_buf_error (line_buf, "unsupported line number version");
      return 0;
    }

  hdrlen = read_offset (line_buf, is_dwarf64);

  hdr_buf = *line_buf;
  hdr_buf.left = hdrlen;

  if (!advance (line_buf, hdrlen))
    return 0;
  
  hdr->min_insn_len = read_byte (&hdr_buf);
  if (hdr->version < 4)
    hdr->max_ops_per_insn = 1;
  else
    hdr->max_ops_per_insn = read_byte (&hdr_buf);

  /* We don't care about default_is_stmt.  */
  read_byte (&hdr_buf);
  
  hdr->line_base = read_sbyte (&hdr_buf);
  hdr->line_range = read_byte (&hdr_buf);

  hdr->opcode_base = read_byte (&hdr_buf);
  hdr->opcode_lengths = hdr_buf.buf;
  if (!advance (&hdr_buf, hdr->opcode_base - 1))
    return 0;

  /* Count the number of directory entries.  */
  hdr->dirs_count = 0;
  p = hdr_buf.buf;
  pend = p + hdr_buf.left;
  while (p < pend && *p != '\0')
    {
      p += strnlen((const char *) p, pend - p) + 1;
      ++hdr->dirs_count;
    }

  hdr->dirs = ((const char **)
	       backtrace_alloc (state,
				hdr->dirs_count * sizeof (const char *),
				line_buf->error_callback, line_buf->data));
  if (hdr->dirs == NULL)
    return 0;

  i = 0;
  while (*hdr_buf.buf != '\0')
    {
      if (hdr_buf.reported_underflow)
	return 0;

      hdr->dirs[i] = (const char *) hdr_buf.buf;
      ++i;
      if (!advance (&hdr_buf,
		    strnlen ((const char *) hdr_buf.buf, hdr_buf.left) + 1))
	return 0;
    }
  if (!advance (&hdr_buf, 1))
    return 0;

  /* Count the number of file entries.  */
  hdr->filenames_count = 0;
  p = hdr_buf.buf;
  pend = p + hdr_buf.left;
  while (p < pend && *p != '\0')
    {
      p += strnlen ((const char *) p, pend - p) + 1;
      p += leb128_len (p);
      p += leb128_len (p);
      p += leb128_len (p);
      ++hdr->filenames_count;
    }

  hdr->filenames = ((const char **)
		    backtrace_alloc (state,
				     hdr->filenames_count * sizeof (char *),
				     line_buf->error_callback,
				     line_buf->data));
  if (hdr->filenames == NULL)
    return 0;
  i = 0;
  while (*hdr_buf.buf != '\0')
    {
      const char *filename;
      uint64_t dir_index;

      if (hdr_buf.reported_underflow)
	return 0;

      filename = (const char *) hdr_buf.buf;
      if (!advance (&hdr_buf,
		    strnlen ((const char *) hdr_buf.buf, hdr_buf.left) + 1))
	return 0;
      dir_index = read_uleb128 (&hdr_buf);
      if (IS_ABSOLUTE_PATH (filename)
	  || (dir_index == 0 && u->comp_dir == NULL))
	hdr->filenames[i] = filename;
      else
	{
	  const char *dir;
	  size_t dir_len;
	  size_t filename_len;
	  char *s;

	  if (dir_index == 0)
	    dir = u->comp_dir;
	  else if (dir_index - 1 < hdr->dirs_count)
	    dir = hdr->dirs[dir_index - 1];
	  else
	    {
	      dwarf_buf_error (line_buf,
			       ("invalid directory index in "
				"line number program header"));
	      return 0;
	    }
	  dir_len = strlen (dir);
	  filename_len = strlen (filename);
	  s = ((char *)
	       backtrace_alloc (state, dir_len + filename_len + 2,
				line_buf->error_callback, line_buf->data));
	  if (s == NULL)
	    return 0;
	  memcpy (s, dir, dir_len);
	  /* FIXME: If we are on a DOS-based file system, and the
	     directory or the file name use backslashes, then we
	     should use a backslash here.  */
	  s[dir_len] = '/';
	  memcpy (s + dir_len + 1, filename, filename_len + 1);
	  hdr->filenames[i] = s;
	}

      /* Ignore the modification time and size.  */
      read_uleb128 (&hdr_buf);
      read_uleb128 (&hdr_buf);

      ++i;
    }

  if (hdr_buf.reported_underflow)
    return 0;

  return 1;
}

/* Read the line program, adding line mappings to VEC.  Return 1 on
   success, 0 on failure.  */

static int
read_line_program (struct backtrace_state *state, struct dwarf_data *ddata,
		   struct unit *u, const struct line_header *hdr,
		   struct dwarf_buf *line_buf, struct line_vector *vec)
{
  uint64_t address;
  unsigned int op_index;
  const char *reset_filename;
  const char *filename;
  int lineno;

  address = 0;
  op_index = 0;
  if (hdr->filenames_count > 0)
    reset_filename = hdr->filenames[0];
  else
    reset_filename = "";
  filename = reset_filename;
  lineno = 1;
  while (line_buf->left > 0)
    {
      unsigned int op;

      op = read_byte (line_buf);
      if (op >= hdr->opcode_base)
	{
	  unsigned int advance;

	  /* Special opcode.  */
	  op -= hdr->opcode_base;
	  advance = op / hdr->line_range;
	  address += (hdr->min_insn_len * (op_index + advance)
		      / hdr->max_ops_per_insn);
	  op_index = (op_index + advance) % hdr->max_ops_per_insn;
	  lineno += hdr->line_base + (int) (op % hdr->line_range);
	  add_line (state, ddata, address, filename, lineno,
		    line_buf->error_callback, line_buf->data, vec);
	}
      else if (op == DW_LNS_extended_op)
	{
	  uint64_t len;

	  len = read_uleb128 (line_buf);
	  op = read_byte (line_buf);
	  switch (op)
	    {
	    case DW_LNE_end_sequence:
	      /* FIXME: Should we mark the high PC here?  It seems
		 that we already have that information from the
		 compilation unit.  */
	      address = 0;
	      op_index = 0;
	      filename = reset_filename;
	      lineno = 1;
	      break;
	    case DW_LNE_set_address:
	      address = read_address (line_buf, u->addrsize);
	      break;
	    case DW_LNE_define_file:
	      {
		const char *f;
		unsigned int dir_index;

		f = (const char *) line_buf->buf;
		if (!advance (line_buf, strnlen (f, line_buf->left) + 1))
		  return 0;
		dir_index = read_uleb128 (line_buf);
		/* Ignore that time and length.  */
		read_uleb128 (line_buf);
		read_uleb128 (line_buf);
		if (IS_ABSOLUTE_PATH (f))
		  filename = f;
		else
		  {
		    const char *dir;
		    size_t dir_len;
		    size_t f_len;
		    char *p;

		    if (dir_index == 0)
		      dir = u->comp_dir;
		    else if (dir_index - 1 < hdr->dirs_count)
		      dir = hdr->dirs[dir_index - 1];
		    else
		      {
			dwarf_buf_error (line_buf,
					 ("invalid directory index "
					  "in line number program"));
			return 0;
		      }
		    dir_len = strlen (dir);
		    f_len = strlen (f);
		    p = ((char *)
			 backtrace_alloc (state, dir_len + f_len + 2,
					  line_buf->error_callback,
					  line_buf->data));
		    if (p == NULL)
		      return 0;
		    memcpy (p, dir, dir_len);
		    /* FIXME: If we are on a DOS-based file system,
		       and the directory or the file name use
		       backslashes, then we should use a backslash
		       here.  */
		    p[dir_len] = '/';
		    memcpy (p + dir_len + 1, f, f_len + 1);
		    filename = p;
		  }
	      }
	      break;
	    case DW_LNE_set_discriminator:
	      /* We don't care about discriminators.  */
	      read_uleb128 (line_buf);
	      break;
	    default:
	      if (!advance (line_buf, len - 1))
		return 0;
	      break;
	    }
	}
      else
	{
	  switch (op)
	    {
	    case DW_LNS_copy:
	      add_line (state, ddata, address, filename, lineno,
			line_buf->error_callback, line_buf->data, vec);
	      break;
	    case DW_LNS_advance_pc:
	      {
		uint64_t advance;

		advance = read_uleb128 (line_buf);
		address += (hdr->min_insn_len * (op_index + advance)
			    / hdr->max_ops_per_insn);
		op_index = (op_index + advance) % hdr->max_ops_per_insn;
	      }
	      break;
	    case DW_LNS_advance_line:
	      lineno += (int) read_sleb128 (line_buf);
	      break;
	    case DW_LNS_set_file:
	      {
		uint64_t fileno;

		fileno = read_uleb128 (line_buf);
		if (fileno == 0)
		  filename = "";
		else
		  {
		    if (fileno - 1 >= hdr->filenames_count)
		      {
			dwarf_buf_error (line_buf,
					 ("invalid file number in "
					  "line number program"));
			return 0;
		      }
		    filename = hdr->filenames[fileno - 1];
		  }
	      }
	      break;
	    case DW_LNS_set_column:
	      read_uleb128 (line_buf);
	      break;
	    case DW_LNS_negate_stmt:
	      break;
	    case DW_LNS_set_basic_block:
	      break;
	    case DW_LNS_const_add_pc:
	      {
		unsigned int advance;

		op = 255 - hdr->opcode_base;
		advance = op / hdr->line_range;
		address += (hdr->min_insn_len * (op_index + advance)
			    / hdr->max_ops_per_insn);
		op_index = (op_index + advance) % hdr->max_ops_per_insn;
	      }
	      break;
	    case DW_LNS_fixed_advance_pc:
	      address += read_uint16 (line_buf);
	      op_index = 0;
	      break;
	    case DW_LNS_set_prologue_end:
	      break;
	    case DW_LNS_set_epilogue_begin:
	      break;
	    case DW_LNS_set_isa:
	      read_uleb128 (line_buf);
	      break;
	    default:
	      {
		unsigned int i;

		for (i = hdr->opcode_lengths[op - 1]; i > 0; --i)
		  read_uleb128 (line_buf);
	      }
	      break;
	    }
	}
    }

  return 1;
}

/* Read the line number information for a compilation unit.  Returns 1
   on success, 0 on failure.  */

static int
read_line_info (struct backtrace_state *state, struct dwarf_data *ddata,
		backtrace_error_callback error_callback, void *data,
		struct unit *u, struct line_header *hdr, struct line **lines,
		size_t *lines_count)
{
  struct line_vector vec;
  struct dwarf_buf line_buf;
  uint64_t len;
  int is_dwarf64;
  struct line *ln;

  memset (&vec.vec, 0, sizeof vec.vec);
  vec.count = 0;

  memset (hdr, 0, sizeof *hdr);

  if (u->lineoff != (off_t) (size_t) u->lineoff
      || (size_t) u->lineoff >= ddata->dwarf_line_size)
    {
      error_callback (data, "unit line offset out of range", 0);
      goto fail;
    }

  line_buf.name = ".debug_line";
  line_buf.start = ddata->dwarf_line;
  line_buf.buf = ddata->dwarf_line + u->lineoff;
  line_buf.left = ddata->dwarf_line_size - u->lineoff;
  line_buf.is_bigendian = ddata->is_bigendian;
  line_buf.error_callback = error_callback;
  line_buf.data = data;
  line_buf.reported_underflow = 0;

  is_dwarf64 = 0;
  len = read_uint32 (&line_buf);
  if (len == 0xffffffff)
    {
      len = read_uint64 (&line_buf);
      is_dwarf64 = 1;
    }
  line_buf.left = len;

  if (!read_line_header (state, u, is_dwarf64, &line_buf, hdr))
    goto fail;

  if (!read_line_program (state, ddata, u, hdr, &line_buf, &vec))
    goto fail;

  if (line_buf.reported_underflow)
    goto fail;

  if (vec.count == 0)
    {
      /* This is not a failure in the sense of a generating an error,
	 but it is a failure in that sense that we have no useful
	 information.  */
      goto fail;
    }

  /* Allocate one extra entry at the end.  */
  ln = ((struct line *)
	backtrace_vector_grow (state, sizeof (struct line), error_callback,
			       data, &vec.vec));
  if (ln == NULL)
    goto fail;
  ln->pc = (uintptr_t) -1;
  ln->filename = NULL;
  ln->lineno = 0;
  ln->idx = 0;

  if (!backtrace_vector_release (state, &vec.vec, error_callback, data))
    goto fail;

  ln = (struct line *) vec.vec.base;
  backtrace_qsort (ln, vec.count, sizeof (struct line), line_compare);

  *lines = ln;
  *lines_count = vec.count;

  return 1;

 fail:
  vec.vec.alc += vec.vec.size;
  vec.vec.size = 0;
  backtrace_vector_release (state, &vec.vec, error_callback, data);
  free_line_header (state, hdr, error_callback, data);
  *lines = (struct line *) (uintptr_t) -1;
  *lines_count = 0;
  return 0;
}

/* Read the name of a function from a DIE referenced by a
   DW_AT_abstract_origin or DW_AT_specification tag.  OFFSET is within
   the same compilation unit.  */

static const char *
read_referenced_name (struct dwarf_data *ddata, struct unit *u,
		      uint64_t offset, backtrace_error_callback error_callback,
		      void *data)
{
  struct dwarf_buf unit_buf;
  uint64_t code;
  const struct abbrev *abbrev;
  const char *ret;
  size_t i;

  /* OFFSET is from the start of the data for this compilation unit.
     U->unit_data is the data, but it starts U->unit_data_offset bytes
     from the beginning.  */

  if (offset < u->unit_data_offset
      || offset - u->unit_data_offset >= u->unit_data_len)
    {
      error_callback (data,
		      "abstract origin or specification out of range",
		      0);
      return NULL;
    }

  offset -= u->unit_data_offset;

  unit_buf.name = ".debug_info";
  unit_buf.start = ddata->dwarf_info;
  unit_buf.buf = u->unit_data + offset;
  unit_buf.left = u->unit_data_len - offset;
  unit_buf.is_bigendian = ddata->is_bigendian;
  unit_buf.error_callback = error_callback;
  unit_buf.data = data;
  unit_buf.reported_underflow = 0;

  code = read_uleb128 (&unit_buf);
  if (code == 0)
    {
      dwarf_buf_error (&unit_buf, "invalid abstract origin or specification");
      return NULL;
    }

  abbrev = lookup_abbrev (&u->abbrevs, code, error_callback, data);
  if (abbrev == NULL)
    return NULL;

  ret = NULL;
  for (i = 0; i < abbrev->num_attrs; ++i)
    {
      struct attr_val val;

      if (!read_attribute (abbrev->attrs[i].form, &unit_buf,
			   u->is_dwarf64, u->version, u->addrsize,
			   ddata->dwarf_str, ddata->dwarf_str_size,
			   &val))
	return NULL;

      switch (abbrev->attrs[i].name)
	{
	case DW_AT_name:
	  /* We prefer the linkage name if get one.  */
	  if (val.encoding == ATTR_VAL_STRING)
	    ret = val.u.string;
	  break;

	case DW_AT_linkage_name:
	case DW_AT_MIPS_linkage_name:
	  if (val.encoding == ATTR_VAL_STRING)
	    return val.u.string;
	  break;

	case DW_AT_specification:
	  if (abbrev->attrs[i].form == DW_FORM_ref_addr
	      || abbrev->attrs[i].form == DW_FORM_ref_sig8)
	    {
	      /* This refers to a specification defined in some other
		 compilation unit.  We can handle this case if we
		 must, but it's harder.  */
	      break;
	    }
	  if (val.encoding == ATTR_VAL_UINT
	      || val.encoding == ATTR_VAL_REF_UNIT)
	    {
	      const char *name;

	      name = read_referenced_name (ddata, u, val.u.uint,
					   error_callback, data);
	      if (name != NULL)
		ret = name;
	    }
	  break;

	default:
	  break;
	}
    }

  return ret;
}

/* Add a single range to U that maps to function.  Returns 1 on
   success, 0 on error.  */

static int
add_function_range (struct backtrace_state *state, struct dwarf_data *ddata,
		    struct function *function, uint64_t lowpc, uint64_t highpc,
		    backtrace_error_callback error_callback,
		    void *data, struct function_vector *vec)
{
  struct function_addrs *p;

  /* Add in the base address here, so that we can look up the PC
     directly.  */
  lowpc += ddata->base_address;
  highpc += ddata->base_address;

  if (vec->count > 0)
    {
      p = (struct function_addrs *) vec->vec.base + vec->count - 1;
      if ((lowpc == p->high || lowpc == p->high + 1)
	  && function == p->function)
	{
	  if (highpc > p->high)
	    p->high = highpc;
	  return 1;
	}
    }

  p = ((struct function_addrs *)
       backtrace_vector_grow (state, sizeof (struct function_addrs),
			      error_callback, data, &vec->vec));
  if (p == NULL)
    return 0;

  p->low = lowpc;
  p->high = highpc;
  p->function = function;
  ++vec->count;
  return 1;
}

/* Add PC ranges to U that map to FUNCTION.  Returns 1 on success, 0
   on error.  */

static int
add_function_ranges (struct backtrace_state *state, struct dwarf_data *ddata,
		     struct unit *u, struct function *function,
		     uint64_t ranges, uint64_t base,
		     backtrace_error_callback error_callback, void *data,
		     struct function_vector *vec)
{
  struct dwarf_buf ranges_buf;

  if (ranges >= ddata->dwarf_ranges_size)
    {
      error_callback (data, "function ranges offset out of range", 0);
      return 0;
    }

  ranges_buf.name = ".debug_ranges";
  ranges_buf.start = ddata->dwarf_ranges;
  ranges_buf.buf = ddata->dwarf_ranges + ranges;
  ranges_buf.left = ddata->dwarf_ranges_size - ranges;
  ranges_buf.is_bigendian = ddata->is_bigendian;
  ranges_buf.error_callback = error_callback;
  ranges_buf.data = data;
  ranges_buf.reported_underflow = 0;

  while (1)
    {
      uint64_t low;
      uint64_t high;

      if (ranges_buf.reported_underflow)
	return 0;

      low = read_address (&ranges_buf, u->addrsize);
      high = read_address (&ranges_buf, u->addrsize);

      if (low == 0 && high == 0)
	break;

      if (is_highest_address (low, u->addrsize))
	base = high;
      else
	{
	  if (!add_function_range (state, ddata, function, low + base,
				   high + base, error_callback, data, vec))
	    return 0;
	}
    }

  if (ranges_buf.reported_underflow)
    return 0;

  return 1;
}

/* Read one entry plus all its children.  Add function addresses to
   VEC.  Returns 1 on success, 0 on error.  */

static int
read_function_entry (struct backtrace_state *state, struct dwarf_data *ddata,
		     struct unit *u, uint64_t base, struct dwarf_buf *unit_buf,
		     const struct line_header *lhdr,
		     backtrace_error_callback error_callback, void *data,
		     struct function_vector *vec_function,
		     struct function_vector *vec_inlined)
{
  while (unit_buf->left > 0)
    {
      uint64_t code;
      const struct abbrev *abbrev;
      int is_function;
      struct function *function;
      struct function_vector *vec;
      size_t i;
      uint64_t lowpc;
      int have_lowpc;
      uint64_t highpc;
      int have_highpc;
      int highpc_is_relative;
      uint64_t ranges;
      int have_ranges;

      code = read_uleb128 (unit_buf);
      if (code == 0)
	return 1;

      abbrev = lookup_abbrev (&u->abbrevs, code, error_callback, data);
      if (abbrev == NULL)
	return 0;

      is_function = (abbrev->tag == DW_TAG_subprogram
		     || abbrev->tag == DW_TAG_entry_point
		     || abbrev->tag == DW_TAG_inlined_subroutine);

      if (abbrev->tag == DW_TAG_inlined_subroutine)
	vec = vec_inlined;
      else
	vec = vec_function;

      function = NULL;
      if (is_function)
	{
	  function = ((struct function *)
		      backtrace_alloc (state, sizeof *function,
				       error_callback, data));
	  if (function == NULL)
	    return 0;
	  memset (function, 0, sizeof *function);
	}

      lowpc = 0;
      have_lowpc = 0;
      highpc = 0;
      have_highpc = 0;
      highpc_is_relative = 0;
      ranges = 0;
      have_ranges = 0;
      for (i = 0; i < abbrev->num_attrs; ++i)
	{
	  struct attr_val val;

	  if (!read_attribute (abbrev->attrs[i].form, unit_buf,
			       u->is_dwarf64, u->version, u->addrsize,
			       ddata->dwarf_str, ddata->dwarf_str_size,
			       &val))
	    return 0;

	  /* The compile unit sets the base address for any address
	     ranges in the function entries.  */
	  if (abbrev->tag == DW_TAG_compile_unit
	      && abbrev->attrs[i].name == DW_AT_low_pc
	      && val.encoding == ATTR_VAL_ADDRESS)
	    base = val.u.uint;

	  if (is_function)
	    {
	      switch (abbrev->attrs[i].name)
		{
		case DW_AT_call_file:
		  if (val.encoding == ATTR_VAL_UINT)
		    {
		      if (val.u.uint == 0)
			function->caller_filename = "";
		      else
			{
			  if (val.u.uint - 1 >= lhdr->filenames_count)
			    {
			      dwarf_buf_error (unit_buf,
					       ("invalid file number in "
						"DW_AT_call_file attribute"));
			      return 0;
			    }
			  function->caller_filename =
			    lhdr->filenames[val.u.uint - 1];
			}
		    }
		  break;

		case DW_AT_call_line:
		  if (val.encoding == ATTR_VAL_UINT)
		    function->caller_lineno = val.u.uint;
		  break;

		case DW_AT_abstract_origin:
		case DW_AT_specification:
		  if (abbrev->attrs[i].form == DW_FORM_ref_addr
		      || abbrev->attrs[i].form == DW_FORM_ref_sig8)
		    {
		      /* This refers to an abstract origin defined in
			 some other compilation unit.  We can handle
			 this case if we must, but it's harder.  */
		      break;
		    }
		  if (val.encoding == ATTR_VAL_UINT
		      || val.encoding == ATTR_VAL_REF_UNIT)
		    {
		      const char *name;

		      name = read_referenced_name (ddata, u, val.u.uint,
						   error_callback, data);
		      if (name != NULL)
			function->name = name;
		    }
		  break;

		case DW_AT_name:
		  if (val.encoding == ATTR_VAL_STRING)
		    {
		      /* Don't override a name we found in some other
			 way, as it will normally be more
			 useful--e.g., this name is normally not
			 mangled.  */
		      if (function->name == NULL)
			function->name = val.u.string;
		    }
		  break;

		case DW_AT_linkage_name:
		case DW_AT_MIPS_linkage_name:
		  if (val.encoding == ATTR_VAL_STRING)
		    function->name = val.u.string;
		  break;

		case DW_AT_low_pc:
		  if (val.encoding == ATTR_VAL_ADDRESS)
		    {
		      lowpc = val.u.uint;
		      have_lowpc = 1;
		    }
		  break;

		case DW_AT_high_pc:
		  if (val.encoding == ATTR_VAL_ADDRESS)
		    {
		      highpc = val.u.uint;
		      have_highpc = 1;
		    }
		  else if (val.encoding == ATTR_VAL_UINT)
		    {
		      highpc = val.u.uint;
		      have_highpc = 1;
		      highpc_is_relative = 1;
		    }
		  break;

		case DW_AT_ranges:
		  if (val.encoding == ATTR_VAL_UINT
		      || val.encoding == ATTR_VAL_REF_SECTION)
		    {
		      ranges = val.u.uint;
		      have_ranges = 1;
		    }
		  break;

		default:
		  break;
		}
	    }
	}

      /* If we couldn't find a name for the function, we have no use
	 for it.  */
      if (is_function && function->name == NULL)
	{
	  backtrace_free (state, function, sizeof *function,
			  error_callback, data);
	  is_function = 0;
	}

      if (is_function)
	{
	  if (have_ranges)
	    {
	      if (!add_function_ranges (state, ddata, u, function, ranges,
					base, error_callback, data, vec))
		return 0;
	    }
	  else if (have_lowpc && have_highpc)
	    {
	      if (highpc_is_relative)
		highpc += lowpc;
	      if (!add_function_range (state, ddata, function, lowpc, highpc,
				       error_callback, data, vec))
		return 0;
	    }
	  else
	    {
	      backtrace_free (state, function, sizeof *function,
			      error_callback, data);
	      is_function = 0;
	    }
	}

      if (abbrev->has_children)
	{
	  if (!is_function)
	    {
	      if (!read_function_entry (state, ddata, u, base, unit_buf, lhdr,
					error_callback, data, vec_function,
					vec_inlined))
		return 0;
	    }
	  else
	    {
	      struct function_vector fvec;

	      /* Gather any information for inlined functions in
		 FVEC.  */

	      memset (&fvec, 0, sizeof fvec);

	      if (!read_function_entry (state, ddata, u, base, unit_buf, lhdr,
					error_callback, data, vec_function,
					&fvec))
		return 0;

	      if (fvec.count > 0)
		{
		  struct function_addrs *faddrs;

		  if (!backtrace_vector_release (state, &fvec.vec,
						 error_callback, data))
		    return 0;

		  faddrs = (struct function_addrs *) fvec.vec.base;
		  backtrace_qsort (faddrs, fvec.count,
				   sizeof (struct function_addrs),
				   function_addrs_compare);

		  function->function_addrs = faddrs;
		  function->function_addrs_count = fvec.count;
		}
	    }
	}
    }

  return 1;
}

/* Read function name information for a compilation unit.  We look
   through the whole unit looking for function tags.  */

static void
read_function_info (struct backtrace_state *state, struct dwarf_data *ddata,
		    const struct line_header *lhdr,
		    backtrace_error_callback error_callback, void *data,
		    struct unit *u, struct function_vector *fvec,
		    struct function_addrs **ret_addrs,
		    size_t *ret_addrs_count)
{
  struct function_vector lvec;
  struct function_vector *pfvec;
  struct dwarf_buf unit_buf;
  struct function_addrs *addrs;
  size_t addrs_count;

  /* Use FVEC if it is not NULL.  Otherwise use our own vector.  */
  if (fvec != NULL)
    pfvec = fvec;
  else
    {
      memset (&lvec, 0, sizeof lvec);
      pfvec = &lvec;
    }

  unit_buf.name = ".debug_info";
  unit_buf.start = ddata->dwarf_info;
  unit_buf.buf = u->unit_data;
  unit_buf.left = u->unit_data_len;
  unit_buf.is_bigendian = ddata->is_bigendian;
  unit_buf.error_callback = error_callback;
  unit_buf.data = data;
  unit_buf.reported_underflow = 0;

  while (unit_buf.left > 0)
    {
      if (!read_function_entry (state, ddata, u, 0, &unit_buf, lhdr,
				error_callback, data, pfvec, pfvec))
	return;
    }

  if (pfvec->count == 0)
    return;

  addrs_count = pfvec->count;

  if (fvec == NULL)
    {
      if (!backtrace_vector_release (state, &lvec.vec, error_callback, data))
	return;
      addrs = (struct function_addrs *) pfvec->vec.base;
    }
  else
    {
      /* Finish this list of addresses, but leave the remaining space in
	 the vector available for the next function unit.  */
      addrs = ((struct function_addrs *)
	       backtrace_vector_finish (state, &fvec->vec,
					error_callback, data));
      if (addrs == NULL)
	return;
      fvec->count = 0;
    }

  backtrace_qsort (addrs, addrs_count, sizeof (struct function_addrs),
		   function_addrs_compare);

  *ret_addrs = addrs;
  *ret_addrs_count = addrs_count;
}

/* See if PC is inlined in FUNCTION.  If it is, print out the inlined
   information, and update FILENAME and LINENO for the caller.
   Returns whatever CALLBACK returns, or 0 to keep going.  */

static int
report_inlined_functions (uintptr_t pc, struct function *function,
			  backtrace_full_callback callback, void *data,
			  const char **filename, int *lineno)
{
  struct function_addrs *function_addrs;
  struct function *inlined;
  int ret;

  if (function->function_addrs_count == 0)
    return 0;

  function_addrs = ((struct function_addrs *)
		    bsearch (&pc, function->function_addrs,
			     function->function_addrs_count,
			     sizeof (struct function_addrs),
			     function_addrs_search));
  if (function_addrs == NULL)
    return 0;

  while (((size_t) (function_addrs - function->function_addrs) + 1
	  < function->function_addrs_count)
	 && pc >= (function_addrs + 1)->low
	 && pc < (function_addrs + 1)->high)
    ++function_addrs;

  /* We found an inlined call.  */

  inlined = function_addrs->function;

  /* Report any calls inlined into this one.  */
  ret = report_inlined_functions (pc, inlined, callback, data,
				  filename, lineno);
  if (ret != 0)
    return ret;

  /* Report this inlined call.  */
  ret = callback (data, pc, *filename, *lineno, inlined->name);
  if (ret != 0)
    return ret;

  /* Our caller will report the caller of the inlined function; tell
     it the appropriate filename and line number.  */
  *filename = inlined->caller_filename;
  *lineno = inlined->caller_lineno;

  return 0;
}

/* Look for a PC in the DWARF mapping for one module.  On success,
   call CALLBACK and return whatever it returns.  On error, call
   ERROR_CALLBACK and return 0.  Sets *FOUND to 1 if the PC is found,
   0 if not.  */

static int
dwarf_lookup_pc (struct backtrace_state *state, struct dwarf_data *ddata,
		 uintptr_t pc, backtrace_full_callback callback,
		 backtrace_error_callback error_callback, void *data,
		 int *found)
{
  struct unit_addrs *entry;
  struct unit *u;
  int new_data;
  struct line *lines;
  struct line *ln;
  struct function_addrs *function_addrs;
  struct function *function;
  const char *filename;
  int lineno;
  int ret;

  *found = 1;

  /* Find an address range that includes PC.  */
  entry = bsearch (&pc, ddata->addrs, ddata->addrs_count,
		   sizeof (struct unit_addrs), unit_addrs_search);

  if (entry == NULL)
    {
      *found = 0;
      return 0;
    }

  /* If there are multiple ranges that contain PC, use the last one,
     in order to produce predictable results.  If we assume that all
     ranges are properly nested, then the last range will be the
     smallest one.  */
  while ((size_t) (entry - ddata->addrs) + 1 < ddata->addrs_count
	 && pc >= (entry + 1)->low
	 && pc < (entry + 1)->high)
    ++entry;

  /* We need the lines, lines_count, function_addrs,
     function_addrs_count fields of u.  If they are not set, we need
     to set them.  When running in threaded mode, we need to allow for
     the possibility that some other thread is setting them
     simultaneously.  */

  u = entry->u;
  lines = u->lines;

  /* Skip units with no useful line number information by walking
     backward.  Useless line number information is marked by setting
     lines == -1.  */
  while (entry > ddata->addrs
	 && pc >= (entry - 1)->low
	 && pc < (entry - 1)->high)
    {
      if (state->threaded)
	lines = (struct line *) backtrace_atomic_load_pointer (&u->lines);

      if (lines != (struct line *) (uintptr_t) -1)
	break;

      --entry;

      u = entry->u;
      lines = u->lines;
    }

  if (state->threaded)
    lines = backtrace_atomic_load_pointer (&u->lines);

  new_data = 0;
  if (lines == NULL)
    {
      size_t function_addrs_count;
      struct line_header lhdr;
      size_t count;

      /* We have never read the line information for this unit.  Read
	 it now.  */

      function_addrs = NULL;
      function_addrs_count = 0;
      if (read_line_info (state, ddata, error_callback, data, entry->u, &lhdr,
			  &lines, &count))
	{
	  struct function_vector *pfvec;

	  /* If not threaded, reuse DDATA->FVEC for better memory
	     consumption.  */
	  if (state->threaded)
	    pfvec = NULL;
	  else
	    pfvec = &ddata->fvec;
	  read_function_info (state, ddata, &lhdr, error_callback, data,
			      entry->u, pfvec, &function_addrs,
			      &function_addrs_count);
	  free_line_header (state, &lhdr, error_callback, data);
	  new_data = 1;
	}

      /* Atomically store the information we just read into the unit.
	 If another thread is simultaneously writing, it presumably
	 read the same information, and we don't care which one we
	 wind up with; we just leak the other one.  We do have to
	 write the lines field last, so that the acquire-loads above
	 ensure that the other fields are set.  */

      if (!state->threaded)
	{
	  u->lines_count = count;
	  u->function_addrs = function_addrs;
	  u->function_addrs_count = function_addrs_count;
	  u->lines = lines;
	}
      else
	{
	  backtrace_atomic_store_size_t (&u->lines_count, count);
	  backtrace_atomic_store_pointer (&u->function_addrs, function_addrs);
	  backtrace_atomic_store_size_t (&u->function_addrs_count,
					 function_addrs_count);
	  backtrace_atomic_store_pointer (&u->lines, lines);
	}
    }

  /* Now all fields of U have been initialized.  */

  if (lines == (struct line *) (uintptr_t) -1)
    {
      /* If reading the line number information failed in some way,
	 try again to see if there is a better compilation unit for
	 this PC.  */
      if (new_data)
	return dwarf_lookup_pc (state, ddata, pc, callback, error_callback,
				data, found);
      return callback (data, pc, NULL, 0, NULL);
    }

  /* Search for PC within this unit.  */

  ln = (struct line *) bsearch (&pc, lines, entry->u->lines_count,
				sizeof (struct line), line_search);
  if (ln == NULL)
    {
      /* The PC is between the low_pc and high_pc attributes of the
	 compilation unit, but no entry in the line table covers it.
	 This implies that the start of the compilation unit has no
	 line number information.  */

      if (entry->u->abs_filename == NULL)
	{
	  const char *filename;

	  filename = entry->u->filename;
	  if (filename != NULL
	      && !IS_ABSOLUTE_PATH (filename)
	      && entry->u->comp_dir != NULL)
	    {
	      size_t filename_len;
	      const char *dir;
	      size_t dir_len;
	      char *s;

	      filename_len = strlen (filename);
	      dir = entry->u->comp_dir;
	      dir_len = strlen (dir);
	      s = (char *) backtrace_alloc (state, dir_len + filename_len + 2,
					    error_callback, data);
	      if (s == NULL)
		{
		  *found = 0;
		  return 0;
		}
	      memcpy (s, dir, dir_len);
	      /* FIXME: Should use backslash if DOS file system.  */
	      s[dir_len] = '/';
	      memcpy (s + dir_len + 1, filename, filename_len + 1);
	      filename = s;
	    }
	  entry->u->abs_filename = filename;
	}

      return callback (data, pc, entry->u->abs_filename, 0, NULL);
    }

  /* Search for function name within this unit.  */

  if (entry->u->function_addrs_count == 0)
    return callback (data, pc, ln->filename, ln->lineno, NULL);

  function_addrs = ((struct function_addrs *)
		    bsearch (&pc, entry->u->function_addrs,
			     entry->u->function_addrs_count,
			     sizeof (struct function_addrs),
			     function_addrs_search));
  if (function_addrs == NULL)
    return callback (data, pc, ln->filename, ln->lineno, NULL);

  /* If there are multiple function ranges that contain PC, use the
     last one, in order to produce predictable results.  */

  while (((size_t) (function_addrs - entry->u->function_addrs + 1)
	  < entry->u->function_addrs_count)
	 && pc >= (function_addrs + 1)->low
	 && pc < (function_addrs + 1)->high)
    ++function_addrs;

  function = function_addrs->function;

  filename = ln->filename;
  lineno = ln->lineno;

  ret = report_inlined_functions (pc, function, callback, data,
				  &filename, &lineno);
  if (ret != 0)
    return ret;

  return callback (data, pc, filename, lineno, function->name);
}


/* Return the file/line information for a PC using the DWARF mapping
   we built earlier.  */

static int
dwarf_fileline (struct backtrace_state *state, uintptr_t pc,
		backtrace_full_callback callback,
		backtrace_error_callback error_callback, void *data)
{
  struct dwarf_data *ddata;
  int found;
  int ret;

  if (!state->threaded)
    {
      for (ddata = (struct dwarf_data *) state->fileline_data;
	   ddata != NULL;
	   ddata = ddata->next)
	{
	  ret = dwarf_lookup_pc (state, ddata, pc, callback, error_callback,
				 data, &found);
	  if (ret != 0 || found)
	    return ret;
	}
    }
  else
    {
      struct dwarf_data **pp;

      pp = (struct dwarf_data **) (void *) &state->fileline_data;
      while (1)
	{
	  ddata = backtrace_atomic_load_pointer (pp);
	  if (ddata == NULL)
	    break;

	  ret = dwarf_lookup_pc (state, ddata, pc, callback, error_callback,
				 data, &found);
	  if (ret != 0 || found)
	    return ret;

	  pp = &ddata->next;
	}
    }

  /* FIXME: See if any libraries have been dlopen'ed.  */

  return callback (data, pc, NULL, 0, NULL);
}

/* Initialize our data structures from the DWARF debug info for a
   file.  Return NULL on failure.  */

static struct dwarf_data *
build_dwarf_data (struct backtrace_state *state,
		  uintptr_t base_address,
		  const unsigned char *dwarf_info,
		  size_t dwarf_info_size,
		  const unsigned char *dwarf_line,
		  size_t dwarf_line_size,
		  const unsigned char *dwarf_abbrev,
		  size_t dwarf_abbrev_size,
		  const unsigned char *dwarf_ranges,
		  size_t dwarf_ranges_size,
		  const unsigned char *dwarf_str,
		  size_t dwarf_str_size,
		  int is_bigendian,
		  backtrace_error_callback error_callback,
		  void *data)
{
  struct unit_addrs_vector addrs_vec;
  struct unit_addrs *addrs;
  size_t addrs_count;
  struct dwarf_data *fdata;

  if (!build_address_map (state, base_address, dwarf_info, dwarf_info_size,
			  dwarf_abbrev, dwarf_abbrev_size, dwarf_ranges,
			  dwarf_ranges_size, dwarf_str, dwarf_str_size,
			  is_bigendian, error_callback, data, &addrs_vec))
    return NULL;

  if (!backtrace_vector_release (state, &addrs_vec.vec, error_callback, data))
    return NULL;
  addrs = (struct unit_addrs *) addrs_vec.vec.base;
  addrs_count = addrs_vec.count;
  backtrace_qsort (addrs, addrs_count, sizeof (struct unit_addrs),
		   unit_addrs_compare);

  fdata = ((struct dwarf_data *)
	   backtrace_alloc (state, sizeof (struct dwarf_data),
			    error_callback, data));
  if (fdata == NULL)
    return NULL;

  fdata->next = NULL;
  fdata->base_address = base_address;
  fdata->addrs = addrs;
  fdata->addrs_count = addrs_count;
  fdata->dwarf_info = dwarf_info;
  fdata->dwarf_info_size = dwarf_info_size;
  fdata->dwarf_line = dwarf_line;
  fdata->dwarf_line_size = dwarf_line_size;
  fdata->dwarf_ranges = dwarf_ranges;
  fdata->dwarf_ranges_size = dwarf_ranges_size;
  fdata->dwarf_str = dwarf_str;
  fdata->dwarf_str_size = dwarf_str_size;
  fdata->is_bigendian = is_bigendian;
  memset (&fdata->fvec, 0, sizeof fdata->fvec);

  return fdata;
}

/* Build our data structures from the DWARF sections for a module.
   Set FILELINE_FN and STATE->FILELINE_DATA.  Return 1 on success, 0
   on failure.  */

int
backtrace_dwarf_add (struct backtrace_state *state,
		     uintptr_t base_address,
		     const unsigned char *dwarf_info,
		     size_t dwarf_info_size,
		     const unsigned char *dwarf_line,
		     size_t dwarf_line_size,
		     const unsigned char *dwarf_abbrev,
		     size_t dwarf_abbrev_size,
		     const unsigned char *dwarf_ranges,
		     size_t dwarf_ranges_size,
		     const unsigned char *dwarf_str,
		     size_t dwarf_str_size,
		     int is_bigendian,
		     backtrace_error_callback error_callback,
		     void *data, fileline *fileline_fn)
{
  struct dwarf_data *fdata;

  fdata = build_dwarf_data (state, base_address, dwarf_info, dwarf_info_size,
			    dwarf_line, dwarf_line_size, dwarf_abbrev,
			    dwarf_abbrev_size, dwarf_ranges, dwarf_ranges_size,
			    dwarf_str, dwarf_str_size, is_bigendian,
			    error_callback, data);
  if (fdata == NULL)
    return 0;

  if (!state->threaded)
    {
      struct dwarf_data **pp;

      for (pp = (struct dwarf_data **) (void *) &state->fileline_data;
	   *pp != NULL;
	   pp = &(*pp)->next)
	;
      *pp = fdata;
    }
  else
    {
      while (1)
	{
	  struct dwarf_data **pp;

	  pp = (struct dwarf_data **) (void *) &state->fileline_data;

	  while (1)
	    {
	      struct dwarf_data *p;

	      p = backtrace_atomic_load_pointer (pp);

	      if (p == NULL)
		break;

	      pp = &p->next;
	    }

	  if (__sync_bool_compare_and_swap (pp, NULL, fdata))
	    break;
	}
    }

  *fileline_fn = dwarf_fileline;

  return 1;
}
