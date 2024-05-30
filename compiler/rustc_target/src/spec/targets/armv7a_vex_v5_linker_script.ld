OUTPUT_FORMAT("elf32-littlearm")

ENTRY(_start)

__user_ram_start = 0x03800000;
__user_ram_length = 0x04800000;
__user_ram_end = __user_ram_start + __user_ram_length;

MEMORY {
    USER_RAM : ORIGIN = __user_ram_start, LENGTH = __user_ram_length
}

__stack_length = 0x400000;
__heap_end = __user_ram_end - __stack_length - 64;

SECTIONS {
    .code_signature __user_ram_start : {
        KEEP(*(.code_signature))
    } > USER_RAM = 0

    .text __user_ram_start + 0x20 : {
        *(.text.boot)
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
        *(.ARM.exidx)
    }
}
