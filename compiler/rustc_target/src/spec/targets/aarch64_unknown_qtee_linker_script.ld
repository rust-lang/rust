ENTRY(tz_app_init);

PHDRS {
  CODE_RO PT_LOAD FILEHDR PHDRS;
  MTDATA PT_LOAD FLAGS(0x04000004);
  GOT PT_LOAD;
  DATA PT_LOAD;
  DYN PT_DYNAMIC;
  REL PT_LOAD;
  RELRO PT_GNU_RELRO;
}

SECTIONS {
 . = SIZEOF_HEADERS;

 .plt : {
    PROVIDE (text_section_base = .);
    * (.plt*)
  } : CODE_RO

  .text : {
    * (.text* kthread_shared)
    KEEP (*(tzapp_syscall_asm))
    * (.rodata*)
  } : CODE_RO

  .eh_frame ALIGN(8) : {
      PROVIDE (__eh_frame = .);
      KEEP (*(.eh_frame)) *(.eh_frame.*)
  }

  PROVIDE (text_section_limit = .);
  PROVIDE (__etext = .);
  PROVIDE (_etext = .);
  PROVIDE (etext = .);

  .metadata ALIGN(4096) : {
    KEEP (*(.metadata))
    KEEP (*(.metadata.*))
  } : MTDATA

  .got ALIGN(4096) : { *(.got.plt) *(.igot.plt) *(.got) *(.igot) } : GOT
    : RELRO

  .data.rel.ro.local : {
    *(.data.rel.ro.local*)
  } : GOT
    : RELRO

  .data.rel.ro : {
    *(.data.rel.ro*)
  } : GOT
    : RELRO

  INIT_ARRAY : {
    FILL(0x0)
    __init_array_start = .;
    *(.init_array)
    __init_array_end = .;
  } : GOT : RELRO

  FINI_ARRAY : {
    FILL(0x0)
    __fini_array_start = .;
    *(.fini_array)
    __fini_array_end = .;
  } : GOT : RELRO

  .data ALIGN(4096) : {
    data_section_base = .;
    * (.data*)
    * (.bss* COMMON)
    data_section_limit = .;
  } : DATA

  .dynamic ALIGN(4096) : { *(.dynamic) } : DYN : REL
  .rela.dyn : { *(.rela.dyn) } : REL
  .rela.plt : { *(.rela.plt) } : REL
  .hash : { *(.hash) } : REL
  .dynsym : { *(.dynsym) } : REL
  .dynstr : { *(.dynstr) } : REL
  .gnu.attributes 0 : { KEEP (*(.gnu.attributes)) }
  .note.gnu.arm.ident 0 : { KEEP (*(.note.gnu.arm.ident)) }
  /DISCARD/ : { *(.ARM.exidx*) *(.note.GNU-stack) *(.gnu_debuglink) *(.gnu.lto_*) *(.init) *(.fini) }
}
