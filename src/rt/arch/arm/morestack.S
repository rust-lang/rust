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
    .align 2
    .type __morestack,%function
__morestack:

    // Save frame pointer and return address
    push {fp, lr}
    
    mov fp, sp

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
    mov sp, fp

    // Save return value
    push {r0, r1}

    // Remove the new allocated stack
    bl upcall_del_stack@plt

    // Restore return value
    pop {r0, r1}

    // Return
    pop {fp, lr}
    mov pc, lr
.endofmorestack:
    .size   __morestack, .endofmorestack-__morestack
   
