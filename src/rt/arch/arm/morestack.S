// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

/* See i386/morestack.S for the lengthy, general explanation. */

.text
.code 32
.arm
.align

.global rust_stack_exhausted
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

    // Create new stack
    bl rust_stack_exhausted@plt

    // the above function ensures that it never returns
    .fnend
