OUTPUT_FORMAT("elf32-littlearm")

ENTRY(_boot)

__user_ram_start = 0x03800000;
__user_ram_length = 0x04800000;
__user_ram_end = __user_ram_start + __user_ram_length;

__code_signature_length = 0x20;

__stack_length = 0x400000;
__heap_end = __user_ram_end - __stack_length;

__eh_frame_hdr_start = SIZEOF(.eh_frame_hdr) > 0 ? ADDR(.eh_frame_hdr) : 0;
__eh_frame_hdr_end = SIZEOF(.eh_frame_hdr) > 0 ? . : 0;

MEMORY {
    USER_RAM : ORIGIN = __user_ram_start, LENGTH = __user_ram_length
}

SECTIONS {
    .code_signature : {
        KEEP(*(.code_signature))
        . = __user_ram_start + __code_signature_length;
    } > USER_RAM

    .text : {
        *(.boot)
        *(.text .text.*)
    } > USER_RAM

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

    /* The unwind tables enabled by "default-uwtable" in the target file live here. */
    /* __eh_frame_start and similar symbols are used by libunwind. */
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

    .heap (NOLOAD) : ALIGN(4) {
        __heap_start = .;
        . = __heap_end;
    } > USER_RAM

    .stack (NOLOAD) : ALIGN(8) {
        __stack_bottom = .;
        . += __stack_length;
        __stack_top = .;
    } > USER_RAM

    /DISCARD/ : {
        *(.ARM.attributes*)
    }
}
