// Do not compile anything here for iOS because split stacks
// are disabled at all and do not need any runtime support.
//
// See also comments in librustrt/stack.rs about why it was
// disabled and how it could be implemented in case of need.
#if !defined(__APPLE__)
// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

.text
.code 32
.arm
.align

#define RECORD_SP_LIMIT record_sp_limit
#define GET_SP_LIMIT    get_sp_limit

.globl 	RECORD_SP_LIMIT
.globl	GET_SP_LIMIT

RECORD_SP_LIMIT:
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
#endif // ANDROID

	str r0, [r3]
	mov pc, lr

GET_SP_LIMIT:
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
#endif  // __ANDROID__

	ldr r0, [r3]
	mov pc, lr
#endif
