OUTPUT_FORMAT("elf32-littlearm")

ENTRY(_start)

__user_ram_start = 0x03800000;
__user_ram_length = 0x04800000;
__user_ram_end = __user_ram_start + __user_ram_length;

MEMORY {
    USER_RAM : ORIGIN = __user_ram_start, LENGTH = __user_ram_length
}

__stack_length = 0x400000;
/*
It's currently unclear why subtracting anything is necessary, but it fixes memory permission errors.
0x100 is an arbitrary number that works.
*/
__heap_end = __user_ram_end - __stack_length - 0x100;

SECTIONS {
    .text : {
        __text_start = .;
        KEEP(*(.code_signature))
        . = __text_start + 0x20;
        *(.boot)
        *(.text .text.*)
    } > USER_RAM = 0

    .rodata1 : {
        *(.rodata1 .rodata1.*)
    } > USER_RAM

    .data : {
        *(.data .data.*)
        *(.data1 .data1.*)
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