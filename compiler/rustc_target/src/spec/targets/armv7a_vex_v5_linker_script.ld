OUTPUT_FORMAT("elf32-littlearm")

ENTRY(_boot)

__user_ram_start = 0x03800000;
__user_ram_length = 0x04800000;
__user_ram_end = __user_ram_start + __user_ram_length;

__code_signature_length = 0x20;

__stack_length = 0x400000;
__heap_end = __user_ram_end - __stack_length;

/* see https://github.com/llvm/llvm-project/blob/main/libunwind/src/AddressSpace.hpp#L78 */
__eh_frame_hdr_start = SIZEOF(.eh_frame_hdr) > 0 ? ADDR(.eh_frame_hdr) : 0;
__eh_frame_hdr_end = SIZEOF(.eh_frame_hdr) > 0 ? . : 0;

MEMORY {
    USER_RAM : ORIGIN = __user_ram_start, LENGTH = __user_ram_length
}

SECTIONS {
    /*
     * VEXos expects program binaries to have a 32-byte header called a "code signature",
     * at their start, which tells the OS that we are a valid program and configures some
     * miscellaneous startup behavior.
     * 
     * This section is then initialized as a weak symbol in our PAL.
     */
    .code_signature : {
        KEEP(*(.code_signature))
        . = __user_ram_start + __code_signature_length;
    } > USER_RAM

    .text : {
        *(.boot)
        *(.text .text.*)
    } > USER_RAM

    /* Global/uninitialized/static/constant data sections. */
    .rodata : {
        *(.rodata .rodata.*)
    } > USER_RAM

    .data : {
        *(.data .data.*)
    } > USER_RAM

    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        __bss_end = .;
    } > USER_RAM

    /*
     * These sections are added by the compiler in some cases to facilitate stack unwinding.
     * __eh_frame_start and similar symbols are used by libunwind.
     */
    .eh_frame_hdr : {
        KEEP(*(.eh_frame_hdr))
    } > USER_RAM

    .eh_frame : {
        __eh_frame_start = .;
       KEEP(*(.eh_frame))
        __eh_frame_end = .;
    } > USER_RAM

    .ARM.exidx : {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > USER_RAM

    .ARM.extab : {
        __extab_start = .;
        *(.ARM.extab*)
        __extab_end = .;
    } > USER_RAM

    /* Active memory regions for the stack/heap. */
    .heap (NOLOAD) : ALIGN(4) {
        __heap_start = .;
        . = __heap_end;
    } > USER_RAM

    .stack (NOLOAD) : ALIGN(8) {
        __stack_bottom = .;
        . += __stack_length;
        __stack_top = .;
    } > USER_RAM

    /*
     * `.ARM.attributes` contains arch metadata for compatibility purposes, but we
     * only target one hardware configuration, meaning it'd just take up space.
     */
    /DISCARD/ : {
        *(.ARM.attributes*)
    }
}
