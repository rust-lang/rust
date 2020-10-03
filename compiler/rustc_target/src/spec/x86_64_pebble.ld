ENTRY(_start)
OUTPUT_FORMAT(elf64-x86-64)

IMAGE_START = 0x10000;

PHDRS {
    text PT_LOAD;
    rodata PT_LOAD FLAGS(4);
    data PT_LOAD;
    caps PT_NOTE;
}

SECTIONS {
    . = IMAGE_START;

    .text : {
        *(.text .text.*)
        . = ALIGN(4K);
    } :text

    .rodata : {
        *(.rodata .rodata.*)
        /* No need to align, because .got is aligned below */
    } :rodata

    .got : {
        *(.got)
        . = ALIGN(4K);
    } :rodata

    .data : {
        *(.data .data.*)
        . = ALIGN(4K);
    } :data

    .bss : {
        *(.bss .bss.*)
        . = ALIGN(4K);
    } :data

    .caps : {
        KEEP(*(.caps))
    } :caps

    /DISCARD/ : {
        *(.eh_frame*)
    }
}
