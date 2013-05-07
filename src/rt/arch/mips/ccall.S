// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

.text

.align 2
.globl __morestack
.hidden __morestack
.cfi_startproc
.set nomips16
.ent __morestack
__morestack:
        .set noreorder
        .set nomacro

        addiu $29, $29, -8
        sw $31, 4($29)
        sw $30, 0($29)

        .cfi_def_cfa_offset 8
        .cfi_offset 31, -4
        .cfi_offset 30, -8

        move $30, $29
        .cfi_def_cfa_register 30

        move $29, $6
        move $25, $5
        jalr $25
        nop
        move $29, $30

        lw $30, 0($29)
        lw $31, 4($29)
        addiu $29, $29, 8

        jr $31
        nop
.end __morestack
.cfi_endproc
