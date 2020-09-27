ENTRY(module_start)
SECTIONS
{
  /* PRX format requires text to begin at 0 */
  .text 0 : { *(.text .text.*) }

  /* Sort stubs for convenient ordering */
  .sceStub.text : { *(.sceStub.text) *(SORT(.sceStub.text.*)) }

  /* Keep these sections around, even though they may appear unused to the linker */
  .lib.ent.top :  { KEEP(*(.lib.ent.top)) }
  .lib.ent :      { KEEP(*(.lib.ent)) }
  .lib.ent.btm :  { KEEP(*(.lib.ent.btm)) }
  .lib.stub.top : { KEEP(*(.lib.stub.top)) }
  .lib.stub :     { KEEP(*(.lib.stub)) }
  .lib.stub.btm : { KEEP(*(.lib.stub.btm)) }
  .eh_frame_hdr : { KEEP(*(.eh_frame_hdr)) }

  /* Add symbols for LLVM's libunwind */
  __eh_frame_hdr_start = SIZEOF(.eh_frame_hdr) > 0 ? ADDR(.eh_frame_hdr) : 0;
  __eh_frame_hdr_end = SIZEOF(.eh_frame_hdr) > 0 ? . : 0;
  .eh_frame :
  {
    __eh_frame_start = .;
    KEEP(*(.eh_frame))
    __eh_frame_end = .;
  }

  /* These are explicitly listed to avoid being merged into .rodata */
  .rodata.sceResident : { *(.rodata.sceResident) }
  .rodata.sceModuleInfo : { *(.rodata.sceModuleInfo) }
  /* Sort NIDs for convenient ordering */
  .rodata.sceNid : { *(.rodata.sceNid) *(SORT(.rodata.sceNid.*)) }
}
