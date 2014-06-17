// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section        .note.GNU-stack, "", @progbits
#endif

/* See i386/morestack.S for the lengthy, general explanation. */

.text

.globl rust_stack_exhausted
.globl __morestack

.hidden __morestack

.cfi_startproc
.set nomips16
.ent __morestack
__morestack:
        .set noreorder
        .set nomacro

        addiu $29, $29, -4
        sw $30, 0($29)

        // 16 = 4 (current) + 12 (previous)
        .cfi_def_cfa_offset 16
        .cfi_offset 31, -4
        .cfi_offset 30, -16

        move $30, $29
        .cfi_def_cfa_register 30

        // O32 ABI always reserves 16 bytes for arguments
        addiu $29, $29, -16

        lw $25, %call16(rust_stack_exhausted)($28)
        jalr $25
        nop

        // the above function make sure that we never get here

.end __morestack
.cfi_endproc
