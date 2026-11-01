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
    KEEP(*(.text.jmp))

    . = 0x80;

    *(.text .text.*)
    *(.plt .plt.*)
  }

  /* Read-only sections */

  . = ALIGN(0x1000);

  .module_name : { *(.module_name) } :rodata

  .rodata : { *(.rodata .rodata.*) } :rodata
  .hash : { *(.hash) }
  .dynsym : { *(.dynsym .dynsym.*) }
  .dynstr : { *(.dynstr .dynstr.*) }
  .rela.dyn : { *(.rela.dyn) }

  .eh_frame : {
    HIDDEN(__eh_frame_start = .);
    *(.eh_frame .eh_frame.*)
    HIDDEN(__eh_frame_end = .);
  }

  .eh_frame_hdr : {
    HIDDEN(__eh_frame_hdr_start = .);
    *(.eh_frame_hdr .eh_frame_hdr.*)
    HIDDEN(__eh_frame_hdr_end = .);
  }

  /* Read-write sections */

   . = ALIGN(0x1000);

  .data : {
    *(.data .data.*)
    *(.got .got.*)
    *(.got.plt .got.plt.*)
  } :data

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
    . = ALIGN(8);
    HIDDEN(__bss_end = .);
  } :bss
}
