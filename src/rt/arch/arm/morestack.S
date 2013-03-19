// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

.text
.code 32
.arm
.align

.global upcall_new_stack
.global upcall_del_stack
.global __morestack
.hidden __morestack

// r4 and r5 are scratch registers for __morestack due to llvm
// ARMFrameLowering::adjustForSegmentedStacks() implementation.
 .type __morestack,%function
__morestack:
	.fnstart
	// Save frame pointer and return address
	.save {r4, r5}
	.save {lr}
	.save {r6, fp, lr}
    push {r6, fp, lr}

	.movsp r6
	mov r6, sp
	.setfp fp, sp, #4
	add fp, sp, #4

    // Save argument registers of the original function
    push {r0, r1, r2, r3, lr}

    mov r0, r4         // The amount of stack needed
    add r1, fp, #20    // Address of stack arguments
    mov r2, r5         // Size of stack arguments
    
    // Create new stack
    bl upcall_new_stack@plt

    // Hold new stack pointer
    mov r5, r0

    // Pop the saved arguments
    pop {r0, r1, r2, r3, lr}

    // Grab the return pointer
    add r4, lr, #16    // Skip past the return
    mov sp, r5         // Swich to the new stack
    mov lr, pc
    mov pc, r4         // Call the original function

    // Switch back to rust stack
    mov sp, r6

    // Save return value
	mov r4, r0
	mov r5, r1

    // Remove the new allocated stack
    bl upcall_del_stack@plt

    // Restore return value
	mov r0, r4
	mov r1, r5
	
    // Return
    pop {r6, fp, lr}
    mov pc, lr
    .fnend
