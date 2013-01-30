.text

.globl __morestack
.hidden __morestack
.align 2
.set nomips16
.ent __morestack
__morestack:
        .set noreorder
        .set nomacro
        move $7, $29
        move $29, $6

        sw $7, 0($29)
        sw $31, -4($29)

        addiu $29, $29, -24
        move $25, $5
        jalr $25
        nop
        addiu $29, $29, 24

        lw $31, -4($29)
        lw $7, 0($29)

        move $29, $7
        jr $31
        nop
.end __morestack
