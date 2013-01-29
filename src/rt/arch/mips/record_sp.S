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
.end get_sp
