// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

.text
.code 32
.arm
#if defined(__APPLE__)
.align 2
#else
.align
#endif

.globl record_sp_limit
.globl get_sp_limit

record_sp_limit:
	// First, try to read TLS address from coprocessor
	mrc p15, #0, r3, c13, c0, #3
	cmp r3, #0
	// Otherwise, try to read from magic address 0xFFFF0FF0
	mvneq r3, #0xF000
	ldreq r3, [r3, #-15]

#if __ANDROID__
	add r3, r3, #252
#elif __linux__
	add r3, r3, #4
#endif

	str r0, [r3]
	mov pc, lr

get_sp_limit:
	// First, try to read TLS address from coprocessor
	mrc p15, #0, r3, c13, c0, #3
	cmp r3, #0
	// Otherwise, try to read from magic address 0xFFFF0FF0
	mvneq r3, #0xF000
	ldreq r3, [r3, #-15]

#if __ANDROID__
	add r3, r3, #252
#elif __linux__
	add r3, r3, #4
#endif

	ldr r0, [r3]
	mov pc, lr
