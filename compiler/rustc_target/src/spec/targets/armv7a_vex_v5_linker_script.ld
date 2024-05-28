OUTPUT_FORMAT("elf32-littlearm")

ENTRY(_start)

__cold_start = 0x03800000;
__cold_length = 0x04800000;
__cold_end = __cold_start + __cold_length;

MEMORY {
    COLD : ORIGIN = __cold_start, LENGTH = __cold_length
}

__stack_length = 0x400000;
/*
I'm not sure why subtracting anything is necessary, but it fixes memory permission errors.
0x1000 is an arbitrary number that works.
*/
__heap_end = __cold_end - __stack_length - 0x1000;

SECTIONS {
    .code_signature {
        KEEP(*(.code_signature))
        . = __cold_start + 0x20;
    } > COLD
    .text : {
        KEEP(*(.text.boot))
        *(.text .text.*)
    } > COLD
    .rodata : {
        __rodata_start = .;
        *(.rodata1 .rodata1.*)
        __rodata_end = .;
    } > COLD
    .data : {
        *(.data .data.*)
        *(.data1 .data1.*)
    } > COLD
    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        __bss_end = .;
    } > COLD
    .heap (NOLOAD) : ALIGN(4) {
        __heap_start = .;
        . = __heap_end;
    } > COLD
    .stack (NOLOAD) : ALIGN(8) {
        __stack_end = .;
        . += __stack_length;
        __stack_start = .;
    } > COLD
}
