// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

.text
.code 32
.arm
.align

.globl __morestack
.hidden __morestack
.type __morestack, %function
__morestack:
	.fnstart
	.save {r4, fp, lr}
	push {r4, fp, lr}
    .movsp r4
	mov r4, sp
	mov sp, r2
	mov fp, sp
	blx r1
	mov sp, r4
	pop {r4, fp, lr}
	mov pc, lr
	.fnend
