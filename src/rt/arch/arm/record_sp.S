// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

.text
.code 32
.arm
.align


.globl record_sp_limit
.globl get_sp_limit
.globl get_sp

record_sp_limit:
	mrc p15, #0, r3, c13, c0, #3
#if __ANDROID__
	add r3, r3, #252
#elif __linux__
	add r3, r3, #4
#endif
	str r0, [r3]
	mov pc, lr

get_sp_limit:
	mrc p15, #0, r3, c13, c0, #3
#if __ANDROID__
	add r3, r3, #252
#elif __linux__
	add r3, r3, #4
#endif
	ldr r0, [r3]
	mov pc, lr

get_sp:
	mov r0, sp
	mov pc, lr
