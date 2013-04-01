// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

.text

.globl record_sp_limit
.align 2
.set nomips16
.ent record_sp_limit
record_sp_limit:
        .set noreorder
        .set nomacro
        .set push
        .set mips32r2
        rdhwr $3, $29
        .set pop
        addiu $3, $3, -0x7008
        sw $4, 4($3)
        jr $31
        nop
.end record_sp_limit

.globl get_sp_limit
.align 2
.set nomips16
.ent get_sp_limit
get_sp_limit:
        .set noreorder
        .set nomacro
        .set push
        .set mips32r2
        rdhwr $3, $29
        .set pop
        addiu $3, $3, -0x7008
        lw $2, 4($3)
        jr $31
        nop
.end get_sp_limit

.globl get_sp
.align 2
.set nomips16
.ent get_sp
get_sp:
        .set noreorder
        .set nomacro
        move $2, $29
        jr $31
        nop
.end get_sp
