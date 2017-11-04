/* macho.c -- Get debug data from an Mach-O file for backtraces.
   Copyright (C) 2012-2016 Free Software Foundation, Inc.
   Written by John Colanduoni.

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

/* We can't use autotools to detect the pointer width of our program because
   we may be building a fat Mach-O file containing both 32-bit and 64-bit
   variants. However Mach-O runs a limited set of platforms so detection
   via preprocessor is not difficult.  */

#if defined(__MACH__)
#if defined(__LP64__)
#define BACKTRACE_BITS 64
#else
#define BACKTRACE_BITS 32
#endif
#else
#error Attempting to build Mach-O support on incorrect platform
#endif

#if defined(__x86_64__)
#define NATIVE_CPU_TYPE CPU_TYPE_X86_64
#elif defined(__i386__)
#define NATIVE_CPU_TYPE CPU_TYPE_X86
#elif defined(__aarch64__)
#define NATIVE_CPU_TYPE CPU_TYPE_ARM64
#elif defined(__arm__)
#define NATIVE_CPU_TYPE CPU_TYPE_ARM
#else
#error Could not detect native Mach-O cpu_type_t
#endif

#include <sys/types.h>
#include <sys/syslimits.h>
#include <string.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach-o/fat.h>
#include <mach-o/dyld.h>
#include <uuid/uuid.h>
#include <dirent.h>
#include <stdlib.h>

#include "backtrace.h"
#include "internal.h"

struct macho_commands_view
{
    struct backtrace_view view;
    uint32_t commands_count;
    uint32_t commands_total_size;
    int bytes_swapped;
    size_t base_offset;
};

enum debug_section
{
    DEBUG_INFO,
    DEBUG_LINE,
    DEBUG_ABBREV,
    DEBUG_RANGES,
    DEBUG_STR,
    DEBUG_MAX
};

static const char *const debug_section_names[DEBUG_MAX] =
    {
        "__debug_info",
        "__debug_line",
        "__debug_abbrev",
        "__debug_ranges",
        "__debug_str"
    };

struct found_dwarf_section
{
    uint32_t file_offset;
    uintptr_t file_size;
    const unsigned char *data;
};

/* Mach-O symbols don't have a length. As a result we have to infer it
   by sorting the symbol addresses for each image and recording the
   memory range attributed to each image.  */
struct macho_symbol
{
    uintptr_t addr;
    size_t size;
    const char *name;
};

struct macho_syminfo_data
{
    struct macho_syminfo_data *next;
    struct macho_symbol *symbols;
    size_t symbol_count;
    uintptr_t min_addr;
    uintptr_t max_addr;
};

uint16_t
macho_file_to_host_u16 (int file_bytes_swapped, uint16_t input)
{
  if (file_bytes_swapped)
    return (input >> 8) | (input << 8);
  else
    return input;
}

uint32_t
macho_file_to_host_u32 (int file_bytes_swapped, uint32_t input)
{
  if (file_bytes_swapped)
    {
      return ((input >> 24) & 0x000000FF)
             | ((input >> 8) & 0x0000FF00)
             | ((input << 8) & 0x00FF0000)
             | ((input << 24) & 0xFF000000);
    }
  else
    {
      return input;
    }
}

uint64_t
macho_file_to_host_u64 (int file_bytes_swapped, uint64_t input)
{
  if (file_bytes_swapped)
    {
      return macho_file_to_host_u32 (file_bytes_swapped,
                                     (uint32_t) (input >> 32))
             | (((uint64_t) macho_file_to_host_u32 (file_bytes_swapped,
                                                    (uint32_t) input)) << 32);
    }
  else
    {
      return input;
    }
}

#if BACKTRACE_BITS == 64
#define macho_file_to_host_usize macho_file_to_host_u64
typedef struct mach_header_64 mach_header_native_t;
#define LC_SEGMENT_NATIVE LC_SEGMENT_64
typedef struct segment_command_64 segment_command_native_t;
typedef struct nlist_64 nlist_native_t;
typedef struct section_64 section_native_t;
#else /* BACKTRACE_BITS == 32 */
#define macho_file_to_host_usize macho_file_to_host_u32
typedef struct mach_header mach_header_native_t;
#define LC_SEGMENT_NATIVE LC_SEGMENT
typedef struct segment_command segment_command_native_t;
typedef struct nlist nlist_native_t;
typedef struct section section_native_t;
#endif

// Gets a view into a Mach-O image, taking any slice offset into account
int
macho_get_view (struct backtrace_state *state, int descriptor,
                off_t offset, size_t size,
                backtrace_error_callback error_callback,
                void *data, struct macho_commands_view *commands_view,
                struct backtrace_view *view)
{
  return backtrace_get_view (state, descriptor,
                             commands_view->base_offset + offset, size,
                             error_callback, data, view);
}

int
macho_get_commands (struct backtrace_state *state, int descriptor,
                    backtrace_error_callback error_callback,
                    void *data, struct macho_commands_view *commands_view,
                    int *incompatible)
{
  int ret = 0;
  int is_fat = 0;
  struct backtrace_view file_header_view;
  int file_header_view_valid = 0;
  struct backtrace_view fat_archs_view;
  int fat_archs_view_valid = 0;
  const mach_header_native_t *file_header;
  uint64_t commands_offset;

  *incompatible = 0;

  if (!backtrace_get_view (state, descriptor, 0, sizeof (mach_header_native_t),
                           error_callback, data, &file_header_view))
    goto end;
  file_header_view_valid = 1;

  switch (*(uint32_t *) file_header_view.data)
    {
      case MH_MAGIC:
        if (BACKTRACE_BITS == 32)
          commands_view->bytes_swapped = 0;
        else
          {
            *incompatible = 1;
            goto end;
          }
      break;
      case MH_CIGAM:
        if (BACKTRACE_BITS == 32)
          commands_view->bytes_swapped = 1;
        else
          {
            *incompatible = 1;
            goto end;
          }
      break;
      case MH_MAGIC_64:
        if (BACKTRACE_BITS == 64)
          commands_view->bytes_swapped = 0;
        else
          {
            *incompatible = 1;
            goto end;
          }
      break;
      case MH_CIGAM_64:
        if (BACKTRACE_BITS == 64)
          commands_view->bytes_swapped = 1;
        else
          {
            *incompatible = 1;
            goto end;
          }
      break;
      case FAT_MAGIC:
        is_fat = 1;
        commands_view->bytes_swapped = 0;
      break;
      case FAT_CIGAM:
        is_fat = 1;
        commands_view->bytes_swapped = 1;
      break;
      default:
        goto end;
    }

  if (is_fat)
    {
      uint32_t native_slice_offset;
      size_t archs_total_size;
      uint32_t arch_count;
      const struct fat_header *fat_header;
      const struct fat_arch *archs;
      uint32_t i;

      fat_header = file_header_view.data;
      arch_count =
          macho_file_to_host_u32 (commands_view->bytes_swapped,
                                  fat_header->nfat_arch);

      archs_total_size = arch_count * sizeof (struct fat_arch);

      if (!backtrace_get_view (state, descriptor, sizeof (fat_header),
                               archs_total_size, error_callback,
                               data, &fat_archs_view))
        goto end;
      fat_archs_view_valid = 1;

      native_slice_offset = 0;
      archs = fat_archs_view.data;
      for (i = 0; i < arch_count; i++)
        {
          const struct fat_arch *raw_arch = archs + i;
          int cpu_type =
              (int) macho_file_to_host_u32 (commands_view->bytes_swapped,
                                            (uint32_t) raw_arch->cputype);

          if (cpu_type == NATIVE_CPU_TYPE)
            {
              native_slice_offset =
                  macho_file_to_host_u32 (commands_view->bytes_swapped,
                                          raw_arch->offset);

              break;
            }
        }

      if (native_slice_offset == 0)
        {
          *incompatible = 1;
          goto end;
        }

      backtrace_release_view (state, &file_header_view, error_callback, data);
      file_header_view_valid = 0;
      if (!backtrace_get_view (state, descriptor, native_slice_offset,
                               sizeof (mach_header_native_t), error_callback,
                               data, &file_header_view))
        goto end;
      file_header_view_valid = 1;

      // The endianess of the slice may be different than the fat image
      switch (*(uint32_t *) file_header_view.data)
        {
          case MH_MAGIC:
            if (BACKTRACE_BITS == 32)
              commands_view->bytes_swapped = 0;
            else
              goto end;
          break;
          case MH_CIGAM:
            if (BACKTRACE_BITS == 32)
              commands_view->bytes_swapped = 1;
            else
              goto end;
          break;
          case MH_MAGIC_64:
            if (BACKTRACE_BITS == 64)
              commands_view->bytes_swapped = 0;
            else
              goto end;
          break;
          case MH_CIGAM_64:
            if (BACKTRACE_BITS == 64)
              commands_view->bytes_swapped = 1;
            else
              goto end;
          break;
          default:
            goto end;
        }

      commands_view->base_offset = native_slice_offset;
    }
  else
    commands_view->base_offset = 0;

  file_header = file_header_view.data;
  commands_view->commands_count =
      macho_file_to_host_u32 (commands_view->bytes_swapped,
                              file_header->ncmds);
  commands_view->commands_total_size =
      macho_file_to_host_u32 (commands_view->bytes_swapped,
                              file_header->sizeofcmds);
  commands_offset =
      commands_view->base_offset + sizeof (mach_header_native_t);

  if (!backtrace_get_view (state, descriptor, commands_offset,
                           commands_view->commands_total_size, error_callback,
                           data, &commands_view->view))
    goto end;

  ret = 1;

end:
  if (file_header_view_valid)
    backtrace_release_view (state, &file_header_view, error_callback, data);
  if (fat_archs_view_valid)
    backtrace_release_view (state, &fat_archs_view, error_callback, data);
  return ret;
}

int
macho_get_uuid (struct backtrace_state *state ATTRIBUTE_UNUSED,
                int descriptor ATTRIBUTE_UNUSED,
                backtrace_error_callback error_callback,
                void *data, struct macho_commands_view *commands_view,
                uuid_t *uuid)
{
  size_t offset = 0;
  uint32_t i = 0;

  for (i = 0; i < commands_view->commands_count; i++)
    {
      const struct load_command *raw_command;
      struct load_command command;

      if (offset + sizeof (struct load_command)
          > commands_view->commands_total_size)
        {
          error_callback (data,
                          "executable file contains out of range command offset",
                          0);
          return 0;
        }

      raw_command =
          commands_view->view.data + offset;
      command.cmd = macho_file_to_host_u32 (commands_view->bytes_swapped,
                                            raw_command->cmd);
      command.cmdsize = macho_file_to_host_u32 (commands_view->bytes_swapped,
                                                raw_command->cmdsize);

      if (command.cmd == LC_UUID)
        {
          const struct uuid_command *uuid_command;

          if (offset + sizeof (struct uuid_command)
              > commands_view->commands_total_size)
            {
              error_callback (data,
                              "executable file contains out of range command offset",
                              0);
              return 0;
            }

          uuid_command =
              (struct uuid_command *) raw_command;
          memcpy (uuid, uuid_command->uuid, sizeof (uuid_t));
          return 1;
        }

      offset += command.cmdsize;
    }

  error_callback (data, "executable file is missing an identifying UUID", 0);
  return 0;
}

/* Returns the base address of a Mach-O image, as encoded in the file header.
 * WARNING: This does not take ASLR into account, which is ubiquitous on recent
 * Darwin platforms.
 */
int
macho_get_addr_range (struct backtrace_state *state ATTRIBUTE_UNUSED,
                      int descriptor ATTRIBUTE_UNUSED,
                      backtrace_error_callback error_callback,
                      void *data, struct macho_commands_view *commands_view,
                      uintptr_t *base_address, uintptr_t *max_address)
{
  size_t offset = 0;
  int found_text = 0;
  uint32_t i = 0;

  *max_address = 0;

  for (i = 0; i < commands_view->commands_count; i++)
    {
      const struct load_command *raw_command;
      struct load_command command;

      if (offset + sizeof (struct load_command)
          > commands_view->commands_total_size)
        {
          error_callback (data,
                          "executable file contains out of range command offset",
                          0);
          return 0;
        }

      raw_command = commands_view->view.data + offset;
      command.cmd = macho_file_to_host_u32 (commands_view->bytes_swapped,
                                            raw_command->cmd);
      command.cmdsize = macho_file_to_host_u32 (commands_view->bytes_swapped,
                                                raw_command->cmdsize);

      if (command.cmd == LC_SEGMENT_NATIVE)
        {
          const segment_command_native_t *raw_segment;
          uintptr_t segment_vmaddr;
          uintptr_t segment_vmsize;
          uintptr_t segment_maxaddr;
          uintptr_t text_fileoff;

          if (offset + sizeof (segment_command_native_t)
              > commands_view->commands_total_size)
            {
              error_callback (data,
                              "executable file contains out of range command offset",
                              0);
              return 0;
            }

          raw_segment = (segment_command_native_t *) raw_command;

          segment_vmaddr = macho_file_to_host_usize (
              commands_view->bytes_swapped, raw_segment->vmaddr);
          segment_vmsize = macho_file_to_host_usize (
              commands_view->bytes_swapped, raw_segment->vmsize);
          segment_maxaddr = segment_vmaddr + segment_vmsize;

          if (strncmp (raw_segment->segname, "__TEXT",
                       sizeof (raw_segment->segname)) == 0)
            {
              text_fileoff = macho_file_to_host_usize (
                  commands_view->bytes_swapped, raw_segment->fileoff);
              *base_address = segment_vmaddr - text_fileoff;

              found_text = 1;
            }

          if (segment_maxaddr > *max_address)
            *max_address = segment_maxaddr;
        }

      offset += command.cmdsize;
    }

  if (found_text)
    return 1;
  else
    {
      error_callback (data, "executable is missing __TEXT segment", 0);
      return 0;
    }
}

static int
macho_symbol_compare_addr (const void *left_raw, const void *right_raw)
{
  const struct macho_symbol *left = left_raw;
  const struct macho_symbol *right = right_raw;

  if (left->addr > right->addr)
    return 1;
  else if (left->addr < right->addr)
    return -1;
  else
    return 0;
}

int
macho_symbol_type_relevant (uint8_t type)
{
  uint8_t type_field = (uint8_t) (type & N_TYPE);

  return !(type & N_EXT) &&
         (type_field == N_ABS || type_field == N_SECT);
}

int
macho_add_symtab (struct backtrace_state *state,
                  backtrace_error_callback error_callback,
                  void *data, int descriptor,
                  struct macho_commands_view *commands_view,
                  uintptr_t base_address, uintptr_t max_image_address,
                  intptr_t vmslide, int *found_sym)
{
  struct macho_syminfo_data *syminfo_data;

  int ret = 0;
  size_t offset = 0;
  struct backtrace_view symtab_view;
  int symtab_view_valid = 0;
  struct backtrace_view strtab_view;
  int strtab_view_valid = 0;
  size_t syminfo_index = 0;
  size_t function_count = 0;
  uint32_t i = 0;
  uint32_t j = 0;
  uint32_t symtab_index = 0;

  *found_sym = 0;

  for (i = 0; i < commands_view->commands_count; i++)
    {
      const struct load_command *raw_command;
      struct load_command command;

      if (offset + sizeof (struct load_command)
          > commands_view->commands_total_size)
        {
          error_callback (data,
                          "executable file contains out of range command offset",
                          0);
          return 0;
        }

      raw_command = commands_view->view.data + offset;
      command.cmd = macho_file_to_host_u32 (commands_view->bytes_swapped,
                                            raw_command->cmd);
      command.cmdsize = macho_file_to_host_u32 (commands_view->bytes_swapped,
                                                raw_command->cmdsize);

      if (command.cmd == LC_SYMTAB)
        {
          const struct symtab_command *symtab_command;
          uint32_t symbol_table_offset;
          uint32_t symbol_count;
          uint32_t string_table_offset;
          uint32_t string_table_size;

          if (offset + sizeof (struct symtab_command)
              > commands_view->commands_total_size)
            {
              error_callback (data,
                              "executable file contains out of range command offset",
                              0);
              return 0;
            }

          symtab_command = (struct symtab_command *) raw_command;

          symbol_table_offset = macho_file_to_host_u32 (
              commands_view->bytes_swapped, symtab_command->symoff);
          symbol_count = macho_file_to_host_u32 (
              commands_view->bytes_swapped, symtab_command->nsyms);
          string_table_offset = macho_file_to_host_u32 (
              commands_view->bytes_swapped, symtab_command->stroff);
          string_table_size = macho_file_to_host_u32 (
              commands_view->bytes_swapped, symtab_command->strsize);


          if (!macho_get_view (state, descriptor, symbol_table_offset,
                               symbol_count * sizeof (nlist_native_t),
                               error_callback, data, commands_view,
                               &symtab_view))
            goto end;
          symtab_view_valid = 1;

          if (!macho_get_view (state, descriptor, string_table_offset,
                               string_table_size, error_callback, data,
                               commands_view, &strtab_view))
            goto end;
          strtab_view_valid = 1;

          // Count functions first
          for (j = 0; j < symbol_count; j++)
            {
              const nlist_native_t *raw_sym =
                  ((const nlist_native_t *) symtab_view.data) + j;

              if (macho_symbol_type_relevant (raw_sym->n_type))
                {
                  function_count += 1;
                }
            }

          // Allocate space for the:
          //  (a) macho_syminfo_data for this image
          //  (b) macho_symbol entries
          syminfo_data =
              backtrace_alloc (state,
                               sizeof (struct macho_syminfo_data),
                               error_callback, data);
          if (syminfo_data == NULL)
            goto end;

          syminfo_data->symbols = backtrace_alloc (
              state, function_count * sizeof (struct macho_symbol),
              error_callback, data);
          if (syminfo_data->symbols == NULL)
            goto end;

          syminfo_data->symbol_count = function_count;
          syminfo_data->next = NULL;
          syminfo_data->min_addr = base_address;
          syminfo_data->max_addr = max_image_address;

          for (symtab_index = 0;
               symtab_index < symbol_count; symtab_index++)
            {
              const nlist_native_t *raw_sym =
                  ((const nlist_native_t *) symtab_view.data) +
                  symtab_index;

              if (macho_symbol_type_relevant (raw_sym->n_type))
                {
                  size_t strtab_index;
                  const char *name;
                  size_t max_len_plus_one;

                  syminfo_data->symbols[syminfo_index].addr =
                      macho_file_to_host_usize (commands_view->bytes_swapped,
                                                raw_sym->n_value) + vmslide;

                  strtab_index = macho_file_to_host_u32 (
                      commands_view->bytes_swapped,
                      raw_sym->n_un.n_strx);

                  // Check the range of the supposed "string" we've been
                  // given
                  if (strtab_index >= string_table_size)
                    {
                      error_callback (
                          data,
                          "dSYM file contains out of range string table index",
                          0);
                      goto end;
                    }

                  name = strtab_view.data + strtab_index;
                  max_len_plus_one = string_table_size - strtab_index;

                  if (strnlen (name, max_len_plus_one) >= max_len_plus_one)
                    {
                      error_callback (
                          data,
                          "dSYM file contains unterminated string",
                          0);
                      goto end;
                    }

                  // Remove underscore prefixes
                  if (name[0] == '_')
                    {
                      name = name + 1;
                    }

                  syminfo_data->symbols[syminfo_index].name = name;

                  syminfo_index += 1;
                }
            }

          backtrace_qsort (syminfo_data->symbols,
                           syminfo_data->symbol_count,
                           sizeof (struct macho_symbol),
                           macho_symbol_compare_addr);

          // Calculate symbol sizes
          for (syminfo_index = 0;
               syminfo_index < syminfo_data->symbol_count; syminfo_index++)
            {
              if (syminfo_index + 1 < syminfo_data->symbol_count)
                {
                  syminfo_data->symbols[syminfo_index].size =
                      syminfo_data->symbols[syminfo_index + 1].addr -
                      syminfo_data->symbols[syminfo_index].addr;
                }
              else
                {
                  syminfo_data->symbols[syminfo_index].size =
                      max_image_address -
                      syminfo_data->symbols[syminfo_index].addr;
                }
            }

          if (!state->threaded)
            {
              struct macho_syminfo_data **pp;

              for (pp = (struct macho_syminfo_data **) (void *) &state->syminfo_data;
                   *pp != NULL;
                   pp = &(*pp)->next);
              *pp = syminfo_data;
            }
          else
            {
              while (1)
                {
                  struct macho_syminfo_data **pp;

                  pp = (struct macho_syminfo_data **) (void *) &state->syminfo_data;

                  while (1)
                    {
                      struct macho_syminfo_data *p;

                      p = backtrace_atomic_load_pointer (pp);

                      if (p == NULL)
                        break;

                      pp = &p->next;
                    }

                  if (__sync_bool_compare_and_swap (pp, NULL, syminfo_data))
                    break;
                }
            }

          strtab_view_valid = 0; // We need to keep string table around
          *found_sym = 1;
          ret = 1;
          goto end;
        }

      offset += command.cmdsize;
    }

  // No symbol table here
  ret = 1;
  goto end;

end:
  if (symtab_view_valid)
    backtrace_release_view (state, &symtab_view, error_callback, data);
  if (strtab_view_valid)
    backtrace_release_view (state, &strtab_view, error_callback, data);
  return ret;
}

int
macho_try_dwarf (struct backtrace_state *state,
                 backtrace_error_callback error_callback,
                 void *data, fileline *fileline_fn, uuid_t *executable_uuid,
                 uintptr_t base_address, uintptr_t max_image_address,
                 intptr_t vmslide, char *dwarf_filename, int *matched,
                 int *found_sym, int *found_dwarf)
{
  uuid_t dwarf_uuid;

  int ret = 0;
  int dwarf_descriptor;
  int dwarf_descriptor_valid = 0;
  struct macho_commands_view commands_view;
  int commands_view_valid = 0;
  struct backtrace_view dwarf_view;
  int dwarf_view_valid = 0;
  size_t offset = 0;
  struct found_dwarf_section dwarf_sections[DEBUG_MAX];
  uintptr_t min_dwarf_offset = 0;
  uintptr_t max_dwarf_offset = 0;
  uint32_t i = 0;
  uint32_t j = 0;
  int k = 0;

  *matched = 0;
  *found_sym = 0;
  *found_dwarf = 0;

  if ((dwarf_descriptor = backtrace_open (dwarf_filename, error_callback,
                                          data, NULL)) == 0)
    goto end;
  dwarf_descriptor_valid = 1;

  int incompatible;
  if (!macho_get_commands (state, dwarf_descriptor, error_callback, data,
                           &commands_view, &incompatible))
    {
      // Failing to read the header here is fine, because this dSYM may be
      // for a different architecture
      if (incompatible)
        {
          ret = 1;
        }
      goto end;
    }
  commands_view_valid = 1;

  // Get dSYM UUID and compare
  if (!macho_get_uuid (state, dwarf_descriptor, error_callback, data,
                       &commands_view, &dwarf_uuid))
    {
      error_callback (data, "dSYM file is missing an identifying uuid", 0);
      goto end;
    }
  if (memcmp (executable_uuid, &dwarf_uuid, sizeof (uuid_t)) != 0)
    {
      // DWARF doesn't belong to desired executable
      ret = 1;
      goto end;
    }

  *matched = 1;

  // Read symbol table
  if (!macho_add_symtab (state, error_callback, data, dwarf_descriptor,
                         &commands_view, base_address, max_image_address,
                         vmslide, found_sym))
    goto end;

  // Get DWARF sections

  memset (dwarf_sections, 0, sizeof (dwarf_sections));
  offset = 0;
  for (i = 0; i < commands_view.commands_count; i++)
    {
      const struct load_command *raw_command;
      struct load_command command;

      if (offset + sizeof (struct load_command)
          > commands_view.commands_total_size)
        {
          error_callback (data,
                          "dSYM file contains out of range command offset", 0);
          goto end;
        }

      raw_command = commands_view.view.data + offset;
      command.cmd = macho_file_to_host_u32 (commands_view.bytes_swapped,
                                            raw_command->cmd);
      command.cmdsize = macho_file_to_host_u32 (commands_view.bytes_swapped,
                                                raw_command->cmdsize);

      if (command.cmd == LC_SEGMENT_NATIVE)
        {
          uint32_t section_count;
          size_t section_offset;
          const segment_command_native_t *raw_segment;

          if (offset + sizeof (segment_command_native_t)
              > commands_view.commands_total_size)
            {
              error_callback (data,
                              "dSYM file contains out of range command offset",
                              0);
              goto end;
            }

          raw_segment = (const segment_command_native_t *) raw_command;

          if (strncmp (raw_segment->segname, "__DWARF",
                       sizeof (raw_segment->segname)) == 0)
            {
              section_count = macho_file_to_host_u32 (
                  commands_view.bytes_swapped,
                  raw_segment->nsects);

              section_offset = offset + sizeof (segment_command_native_t);

              // Search sections for relevant DWARF section names
              for (j = 0; j < section_count; j++)
                {
                  const section_native_t *raw_section;

                  if (section_offset + sizeof (section_native_t) >
                      commands_view.commands_total_size)
                    {
                      error_callback (data,
                                      "dSYM file contains out of range command offset",
                                      0);
                      goto end;
                    }

                  raw_section = commands_view.view.data + section_offset;

                  for (k = 0; k < DEBUG_MAX; k++)
                    {
                      uintptr_t dwarf_section_end;

                      if (strncmp (raw_section->sectname,
                                   debug_section_names[k],
                                   sizeof (raw_section->sectname)) == 0)
                        {
                          *found_dwarf = 1;

                          dwarf_sections[k].file_offset =
                              macho_file_to_host_u32 (
                                  commands_view.bytes_swapped,
                                  raw_section->offset);
                          dwarf_sections[k].file_size =
                              macho_file_to_host_usize (
                                  commands_view.bytes_swapped,
                                  raw_section->size);

                          if (min_dwarf_offset == 0 ||
                              dwarf_sections[k].file_offset <
                              min_dwarf_offset)
                            min_dwarf_offset = dwarf_sections[k].file_offset;

                          dwarf_section_end =
                              dwarf_sections[k].file_offset +
                              dwarf_sections[k].file_size;
                          if (dwarf_section_end > max_dwarf_offset)
                            max_dwarf_offset = dwarf_section_end;

                          break;
                        }
                    }

                  section_offset += sizeof (section_native_t);
                }

              break;
            }
        }

      offset += command.cmdsize;
    }

  if (!*found_dwarf)
    {
      // No DWARF in this file
      ret = 1;
      goto end;
    }

  if (!macho_get_view (state, dwarf_descriptor, (off_t) min_dwarf_offset,
                       max_dwarf_offset - min_dwarf_offset, error_callback,
                       data, &commands_view, &dwarf_view))
    goto end;
  dwarf_view_valid = 1;

  for (i = 0; i < DEBUG_MAX; i++)
    {
      if (dwarf_sections[i].file_offset == 0)
        dwarf_sections[i].data = NULL;
      else
        dwarf_sections[i].data =
            dwarf_view.data + dwarf_sections[i].file_offset - min_dwarf_offset;
    }

  if (!backtrace_dwarf_add (state, vmslide,
                            dwarf_sections[DEBUG_INFO].data,
                            dwarf_sections[DEBUG_INFO].file_size,
                            dwarf_sections[DEBUG_LINE].data,
                            dwarf_sections[DEBUG_LINE].file_size,
                            dwarf_sections[DEBUG_ABBREV].data,
                            dwarf_sections[DEBUG_ABBREV].file_size,
                            dwarf_sections[DEBUG_RANGES].data,
                            dwarf_sections[DEBUG_RANGES].file_size,
                            dwarf_sections[DEBUG_STR].data,
                            dwarf_sections[DEBUG_STR].file_size,
                            ((__DARWIN_BYTE_ORDER == __DARWIN_BIG_ENDIAN)
                            ^ commands_view.bytes_swapped),
                            error_callback, data, fileline_fn))
    goto end;

  // Don't release the DWARF view because it is still in use
  dwarf_descriptor_valid = 0;
  dwarf_view_valid = 0;
  ret = 1;

end:
  if (dwarf_descriptor_valid)
    backtrace_close (dwarf_descriptor, error_callback, data);
  if (commands_view_valid)
    backtrace_release_view (state, &commands_view.view, error_callback, data);
  if (dwarf_view_valid)
    backtrace_release_view (state, &dwarf_view, error_callback, data);
  return ret;
}

int
macho_try_dsym (struct backtrace_state *state,
                backtrace_error_callback error_callback,
                void *data, fileline *fileline_fn, uuid_t *executable_uuid,
                uintptr_t base_address, uintptr_t max_image_address,
                intptr_t vmslide, char *dsym_filename, int *matched,
                int *found_sym, int *found_dwarf)
{
  int ret = 0;
  char dwarf_image_dir_path[PATH_MAX];
  DIR *dwarf_image_dir;
  int dwarf_image_dir_valid = 0;
  struct dirent *directory_entry;
  char dwarf_filename[PATH_MAX];
  int dwarf_matched;
  int dwarf_had_sym;
  int dwarf_had_dwarf;

  *matched = 0;
  *found_sym = 0;
  *found_dwarf = 0;

  strncpy (dwarf_image_dir_path, dsym_filename, PATH_MAX);
  strncat (dwarf_image_dir_path, "/Contents/Resources/DWARF", PATH_MAX);

  if (!(dwarf_image_dir = opendir (dwarf_image_dir_path)))
    {
      error_callback (data, "could not open DWARF directory in dSYM",
                      0);
      goto end;
    }
  dwarf_image_dir_valid = 1;

  while ((directory_entry = readdir (dwarf_image_dir)))
    {
      if (directory_entry->d_type != DT_REG)
        continue;

      strncpy (dwarf_filename, dwarf_image_dir_path, PATH_MAX);
      strncat (dwarf_filename, "/", PATH_MAX);
      strncat (dwarf_filename, directory_entry->d_name, PATH_MAX);

      if (!macho_try_dwarf (state, error_callback, data, fileline_fn,
                            executable_uuid, base_address, max_image_address,
                            vmslide, dwarf_filename,
                            &dwarf_matched, &dwarf_had_sym, &dwarf_had_dwarf))
        goto end;

      if (dwarf_matched)
        {
          *matched = 1;
          *found_sym = dwarf_had_sym;
          *found_dwarf = dwarf_had_dwarf;
          ret = 1;
          goto end;
        }
    }

  // No matching DWARF in this dSYM
  ret = 1;
  goto end;

end:
  if (dwarf_image_dir_valid)
    closedir (dwarf_image_dir);
  return ret;
}

int
macho_add (struct backtrace_state *state,
           backtrace_error_callback error_callback, void *data, int descriptor,
           const char *filename, fileline *fileline_fn, intptr_t vmslide,
           int *found_sym, int *found_dwarf)
{
  uuid_t image_uuid;
  uintptr_t image_file_base_address;
  uintptr_t image_file_max_address;
  uintptr_t image_actual_base_address = 0;
  uintptr_t image_actual_max_address = 0;

  int ret = 0;
  struct macho_commands_view commands_view;
  int commands_view_valid = 0;
  char executable_dirname[PATH_MAX];
  size_t filename_len;
  DIR *executable_dir = NULL;
  int executable_dir_valid = 0;
  struct dirent *directory_entry;
  char dsym_full_path[PATH_MAX];
  static const char *extension;
  size_t extension_len;
  ssize_t i;

  *found_sym = 0;
  *found_dwarf = 0;

  // Find Mach-O commands list
  int incompatible;
  if (!macho_get_commands (state, descriptor, error_callback, data,
                           &commands_view, &incompatible))
    goto end;
  commands_view_valid = 1;

  // First we need to get the uuid of our file so we can hunt down the correct
  // dSYM
  if (!macho_get_uuid (state, descriptor, error_callback, data, &commands_view,
                       &image_uuid))
    goto end;

  // Now we need to find the in memory base address. Step one is to find out
  // what the executable thinks the base address is
  if (!macho_get_addr_range (state, descriptor, error_callback, data,
                             &commands_view,
                             &image_file_base_address,
                             &image_file_max_address))
    goto end;

  image_actual_base_address =
      image_file_base_address + vmslide;
  image_actual_max_address =
      image_file_max_address + vmslide;

  if (image_actual_base_address == 0)
    {
      error_callback (data, "executable file is not loaded", 0);
      goto end;
    }

  // Look for dSYM in our executable's directory
  strncpy (executable_dirname, filename, PATH_MAX);
  filename_len = strlen (executable_dirname);
  for (i = filename_len - 1; i >= 0; i--)
    {
      if (executable_dirname[i] == '/')
        {
          executable_dirname[i] = '\0';
          break;
        }
      else if (i == 0)
        {
          executable_dirname[0] = '.';
          executable_dirname[1] = '\0';
          break;
        }
    }

  if (!(executable_dir = opendir (executable_dirname)))
    {
      error_callback (data, "could not open directory containing executable",
                      0);
      goto end;
    }
  executable_dir_valid = 1;

  extension = ".dSYM";
  extension_len = strlen (extension);
  while ((directory_entry = readdir (executable_dir)))
    {
      if (directory_entry->d_namlen < extension_len)
        continue;
      if (strncasecmp (directory_entry->d_name + directory_entry->d_namlen
                       - extension_len, extension, extension_len) == 0)
        {
          int matched;
          int dsym_had_sym;
          int dsym_had_dwarf;

          // Found a dSYM
          strncpy (dsym_full_path, executable_dirname, PATH_MAX);
          strncat (dsym_full_path, "/", PATH_MAX);
          strncat (dsym_full_path, directory_entry->d_name, PATH_MAX);

          if (!macho_try_dsym (state, error_callback, data,
                               fileline_fn, &image_uuid,
                               image_actual_base_address,
                               image_actual_max_address, vmslide,
                               dsym_full_path,
                               &matched, &dsym_had_sym, &dsym_had_dwarf))
            goto end;

          if (matched)
            {
              *found_sym = dsym_had_sym;
              *found_dwarf = dsym_had_dwarf;
              ret = 1;
              goto end;
            }
        }
    }

  // No matching dSYM
  ret = 1;
  goto end;

end:
  if (commands_view_valid)
    backtrace_release_view (state, &commands_view.view, error_callback,
                            data);
  if (executable_dir_valid)
    closedir (executable_dir);
  return ret;
}

static int
macho_symbol_search (const void *vkey, const void *ventry)
{
  const uintptr_t *key = (const uintptr_t *) vkey;
  const struct macho_symbol *entry = (const struct macho_symbol *) ventry;
  uintptr_t addr;

  addr = *key;
  if (addr < entry->addr)
    return -1;
  else if (addr >= entry->addr + entry->size)
    return 1;
  else
    return 0;
}

static void
macho_syminfo (struct backtrace_state *state,
               uintptr_t addr,
               backtrace_syminfo_callback callback,
               backtrace_error_callback error_callback ATTRIBUTE_UNUSED,
               void *data)
{
  struct macho_syminfo_data *edata;
  struct macho_symbol *sym = NULL;

  if (!state->threaded)
    {
      for (edata = (struct macho_syminfo_data *) state->syminfo_data;
           edata != NULL;
           edata = edata->next)
        {
          if (addr >= edata->min_addr && addr <= edata->max_addr)
            {
              sym = ((struct macho_symbol *)
                  bsearch (&addr, edata->symbols, edata->symbol_count,
                           sizeof (struct macho_symbol), macho_symbol_search));
              if (sym != NULL)
                break;
            }
        }
    }
  else
    {
      struct macho_syminfo_data **pp;

      pp = (struct macho_syminfo_data **) (void *) &state->syminfo_data;
      while (1)
        {
          edata = backtrace_atomic_load_pointer (pp);
          if (edata == NULL)
            break;

          if (addr >= edata->min_addr && addr <= edata->max_addr)
            {
              sym = ((struct macho_symbol *)
                  bsearch (&addr, edata->symbols, edata->symbol_count,
                           sizeof (struct macho_symbol), macho_symbol_search));
              if (sym != NULL)
                break;
            }

          pp = &edata->next;
        }
    }

  if (sym == NULL)
    callback (data, addr, NULL, 0, 0);
  else
    callback (data, addr, sym->name, sym->addr, sym->size);
}


static int
macho_nodebug (struct backtrace_state *state ATTRIBUTE_UNUSED,
               uintptr_t pc ATTRIBUTE_UNUSED,
               backtrace_full_callback callback ATTRIBUTE_UNUSED,
               backtrace_error_callback error_callback, void *data)
{
  error_callback (data, "no debug info in Mach-O executable", -1);
  return 0;
}

static void
macho_nosyms (struct backtrace_state *state ATTRIBUTE_UNUSED,
              uintptr_t addr ATTRIBUTE_UNUSED,
              backtrace_syminfo_callback callback ATTRIBUTE_UNUSED,
              backtrace_error_callback error_callback, void *data)
{
  error_callback (data, "no symbol table in Mach-O executable", -1);
}

int
backtrace_initialize (struct backtrace_state *state, int descriptor,
                      backtrace_error_callback error_callback,
                      void *data, fileline *fileline_fn)
{
  int ret;
  fileline macho_fileline_fn = macho_nodebug;
  int found_sym = 0;
  int found_dwarf = 0;
  uint32_t i = 0;
  uint32_t loaded_image_count;

  // Add all loaded images
  loaded_image_count = _dyld_image_count ();
  for (i = 0; i < loaded_image_count; i++)
    {
      int current_found_sym;
      int current_found_dwarf;
      int current_descriptor;
      intptr_t current_vmslide;
      const char *current_name;

      current_vmslide = _dyld_get_image_vmaddr_slide (i);
      current_name = _dyld_get_image_name (i);

      if (current_name == NULL)
        continue;

      if (!(current_descriptor =
                backtrace_open (current_name, error_callback, data, NULL)))
        {
          continue;
        }

      if (!macho_add (state, error_callback, data, current_descriptor,
                      current_name, &macho_fileline_fn, current_vmslide,
                      &current_found_sym, &current_found_dwarf))
        {
          return 0;
        }

      backtrace_close (current_descriptor, error_callback, data);
      found_sym = found_sym || current_found_sym;
      found_dwarf = found_dwarf || current_found_dwarf;
    }

  if (!state->threaded)
    {
      if (found_sym)
        state->syminfo_fn = macho_syminfo;
      else if (state->syminfo_fn == NULL)
        state->syminfo_fn = macho_nosyms;
    }
  else
    {
      if (found_sym)
        backtrace_atomic_store_pointer (&state->syminfo_fn, macho_syminfo);
      else
        (void) __sync_bool_compare_and_swap (&state->syminfo_fn, NULL,
                                             macho_nosyms);
    }

  if (!state->threaded)
    {
      if (state->fileline_fn == NULL || state->fileline_fn == macho_nodebug)
        *fileline_fn = macho_fileline_fn;
    }
  else
    {
      fileline current_fn;

      current_fn = backtrace_atomic_load_pointer (&state->fileline_fn);
      if (current_fn == NULL || current_fn == macho_nodebug)
        *fileline_fn = macho_fileline_fn;
    }

  return 1;
}

