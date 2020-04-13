OUTPUT_FORMAT(elf64-littleaarch64)
OUTPUT_ARCH(aarch64)
ENTRY(_start)

PHDRS
{
  text PT_LOAD FLAGS(5);
  rodata PT_LOAD FLAGS(4);
  data PT_LOAD FLAGS(6);
  bss PT_LOAD FLAGS(6);
  dynamic PT_DYNAMIC;
}

SECTIONS
{
  . = 0;

  .text : ALIGN(0x1000) {
    HIDDEN(__text_start = .);
    *(.nro_header)
    KEEP(*(.text.jmp))

    . = 0x80;

    *(.text .text.*)
    *(.plt .plt.*)
    HIDDEN(__text_end = .);
  } :text

  /* Read-only sections */


  . = ALIGN(0x1000);
  .module_name : { KEEP (*(.rodata.module_name)) } :rodata 
  
  .hooks : {
      __hook_array_start = .;
      *(.rodata.hooks)
      __hook_array_end = .;
  } :rodata
  
  .rodata : {
    *(.rodata .rodata.*) 
  } :rodata

  .mod0 : {
    KEEP(crt0.nso.o(.data.mod0))
    KEEP(crt0.nro.o(.data.mod0))
    KEEP(crt0.lib.nro.o(.data.mod0))
  }
  .hash : { *(.hash) }
  .dynsym : { *(.dynsym .dynsym.*) }
  .dynstr : { *(.dynstr .dynstr.*) }
  .rela.dyn : { *(.rela.dyn) }

  .eh_frame : ONLY_IF_RO {
    HIDDEN(__eh_frame_start = .);
    KEEP (*(.eh_frame)) *(.eh_frame.*)
    HIDDEN(__eh_frame_end = .);
  }

  .eh_frame_hdr : {
    HIDDEN(__eh_frame_hdr_start = .);
    *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*)
    HIDDEN(__eh_frame_hdr_end = .);
  }
  .gcc_except_table : ONLY_IF_RO { *(.gcc_except_table .gcc_except_table.*) }
  .gnu_extab : ONLY_IF_RO { *(.gnu_extab*) }

  /* Read-write sections */

   . = ALIGN(0x1000);

  .data : {
    *(.data .data.*)
    *(.got .got.*)
    *(.got.plt .got.plt.*)
  } :data

  .eh_frame : ONLY_IF_RW {
    HIDDEN(__eh_frame_start = .);
    KEEP (*(.eh_frame)) *(.eh_frame.*)
    HIDDEN(__eh_frame_end = .);
  }
  .gnu_extab : ONLY_IF_RW { *(.gnu_extab*) }
  .gcc_except_table : ONLY_IF_RW { *(.gcc_except_table .gcc_except_table.*) }

  .dynamic : {
    HIDDEN(__dynamic_start = .);
    *(.dynamic)
  }

  /* BSS section */

  . = ALIGN(0x1000);

  .bss : {
    HIDDEN(__bss_start = .);
    *(.bss .bss.*)
    *(COMMON)
    HIDDEN(__nx_module_runtime = .);
    *(.bss.module_runtime)
    . = ALIGN(8);
    HIDDEN(__bss_end = .);
  } :bss
}
