OUTPUT_FORMAT("elf32-littlearm")
ENTRY(_boot)

/*
 * PROVIDE() is used here so that users can override default values.
 * This is intended to give developers the option to use this Rust
 * target even if the default values in this linker script aren't
 * suitable for their needs.
 *
 * For example: `-C link-arg=--defsym=__stack_length=8M` could
 * be used to increase the stack size above the value set in this
 * file.
 */

PROVIDE(__vcodesig_magic = 0x35585658);     /* XVX5                 */
PROVIDE(__vcodesig_type = 0);               /* V5_SIG_TYPE_USER     */
PROVIDE(__vcodesig_owner = 2);              /* V5_SIG_OWNER_PARTNER */
PROVIDE(__vcodesig_options = 0);            /* none (0)             */

__user_ram_start = 0x03800000;
__user_ram_end   = 0x08000000;
/* (0x48 =) 72 MiB length */
__user_ram_length = __user_ram_start - __user_ram_end;

/*
 * VEXos provides a method for pre-loading a "linked file" at a specified
 * address in User RAM, conventionally near the end, after the primary
 * program binary. We need to be sure not to place any data in that location,
 * so we allow the user of this linker script to inform the start address of
 * this blob.
 */
PROVIDE(__linked_file_length = 0);
PROVIDE(__linked_file_end = __user_ram_end);
PROVIDE(__linked_file_start = __linked_file_end - __linked_file_length);

PROVIDE(__stack_length = 4M);
PROVIDE(__stack_top = __linked_file_start);
PROVIDE(__stack_bottom = __linked_file_start - __stack_length);

MEMORY {
    USER_RAM (RWX) : ORIGIN = __user_ram_start, LENGTH = __user_ram_length
}

SECTIONS {
    /*
     * VEXos expects program binaries to have a 32-byte header called a "code signature"
     * at their start which tells the OS that we are a valid program and configures some
     * miscellaneous startup behavior.
     */
    .code_signature : {
        LONG(__vcodesig_magic)
        LONG(__vcodesig_type)
        LONG(__vcodesig_owner)
        LONG(__vcodesig_options)

        FILL(0)
        . = __user_ram_start + 0x20;
    } > USER_RAM

    /*
     * Executable program instructions.
     */
    .text : {
        /* _boot routine (entry point from VEXos, must be at 0x03800020) */
        *(.boot)

        /* The rest of the program. */
        *(.text .text.*)
    } > USER_RAM

    /*
     * Global/uninitialized/static/constant data sections.
     */
    .rodata : {
        *(.rodata .rodata1 .rodata.*)
        *(.srodata .srodata.*)
    } > USER_RAM

    /*
     * ARM Stack Unwinding Sections
     *
     * These sections are added by the compiler in some cases to facilitate stack unwinding.
     * __eh_frame_start and similar symbols are used by libunwind.
     */

    .except_ordered : {
        PROVIDE(__extab_start = .);
        *(.gcc_except_table *.gcc_except_table.*)
        *(.ARM.extab*)
        PROVIDE(__extab_end = .);
    } > USER_RAM

    .eh_frame_hdr : {
        /* see https://github.com/llvm/llvm-project/blob/main/libunwind/src/AddressSpace.hpp#L78 */
        PROVIDE(__eh_frame_hdr_start = .);
        KEEP(*(.eh_frame_hdr))
        PROVIDE(__eh_frame_hdr_end = .);
    } > USER_RAM

    .eh_frame : {
        PROVIDE(__eh_frame_start = .);
        KEEP(*(.eh_frame))
        PROVIDE(__eh_frame_end = .);
    } > USER_RAM

    .except_unordered : {
        PROVIDE(__exidx_start = .);
        *(.ARM.exidx*)
        PROVIDE(__exidx_end = .);
    } > USER_RAM

    /* -- Data intended to be mutable at runtime begins here. -- */

    .data : {
        *(.data .data1 .data.*)
        *(.sdata .sdata.* .sdata2.*)
    } > USER_RAM

    /* -- End of loadable sections - anything beyond this point shouldn't go in the binary uploaded to the device. -- */

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.sbss*)
        *(.bss .bss.*)

        /* Align the heap */
        . = ALIGN(8);
        __bss_end = .;
    } > USER_RAM

    /*
     * Active memory sections for the stack/heap.
     *
     * Because these are (NOLOAD), they will not influence the final size of the binary.
     */
    .heap (NOLOAD) : {
        __heap_start = .;
        . = __stack_bottom;
        __heap_end = .;
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
