// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section        .note.GNU-stack, "", @progbits
#endif

.text

.globl upcall_new_stack
.globl upcall_del_stack
.globl __morestack

.hidden __morestack

.cfi_startproc
.set nomips16
.ent __morestack
__morestack:
        .set noreorder
        .set nomacro

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
        lw $25, %call16(upcall_new_stack)($23)
        jalr $25
        nop

        // Pop the saved arguments
        lw $4, 16($29)
        lw $5, 20($29)
        lw $6, 24($29)
        lw $7, 28($29)
        addiu $29, $29, 32

        lw $24, 8($30)     // Grab the return pointer.
        addiu $24, $24, 12 // Skip past the `lw`, `jr`, `addiu` in our parent frame
        move $29, $2       // Switch to the new stack.

        // for PIC
        lw $2, 12($30)
        lw $25, 16($30)

        move $28, $23
        jalr $24           // Reenter the caller function
        nop

        // Switch back to the rust stack
        move $29, $30

        // Save the return value
        addiu $29, $29, -24
        sw $2, 16($29)
        sw $3, 20($29)

        move $28, $23
        lw $25, %call16(upcall_del_stack)($23)
        jalr $25
        nop

        // Restore the return value
        lw $2, 16($29)
        lw $3, 20($29)
        addiu $29, $29, 24

        lw $31, 8($29)
        lw $30, 4($29)
        lw $23, 0($29)
        addiu $29, $29, 12

        jr $31
        nop
.end __morestack
.cfi_endproc
