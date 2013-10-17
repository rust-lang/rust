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

        // n.b. most of this is probably unnecessary. I know very little mips
        //      assembly, and I didn't have anything to test on, so I wasn't
        //      brave enough to try to trim this down.

        addiu $29, $29, -12
        sw $31, 8($29)
        sw $30, 4($29)
        sw $23, 0($29)

        // 24 = 12 (current) + 12 (previous)
        .cfi_def_cfa_offset 24
        .cfi_offset 31, -4
        .cfi_offset 30, -20
        .cfi_offset 23, -24

        move $23, $28
        move $30, $29
        .cfi_def_cfa_register 30

        // Save argument registers of the original function
        addiu $29, $29, -32
        sw $4, 16($29)
        sw $5, 20($29)
        sw $6, 24($29)
        sw $7, 28($29)

        move $4, $14     // Size of stack arguments
        addu $5, $30, 24 // Address of stack arguments
        move $6, $15     // The amount of stack needed

        move $28, $23
        lw $25, %call16(rust_stack_exhausted)($23)
        jalr $25
        nop

        // the above function make sure that we never get here

.end __morestack
.cfi_endproc
