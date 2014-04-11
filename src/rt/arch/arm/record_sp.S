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

.globl _record_sp_limit
.globl _get_sp_limit

        // Apple implementation for now is absolutely empty
        // as there is no hardware TLS and emulating
        // it is too expensive for every function call

        // NOTE: although there is Android and Linux
        // implementations, they are actually for kernel mode
        // only, i.e. if you plan to use rust in user space
        // you'll need probably disable them

_record_sp_limit:
#if !defined(__APPLE__)
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
#endif // __APPLE__
	mov pc, lr

_get_sp_limit:
#if defined(__APPLE__)
        // just an placeholder, considering 512Kb limit for stack on secondary thread
        // anyway, stacked segments are disabled for iOS so this function
        // shouldn't be called at all
        mov r0, 65536
#else
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
#endif // __APPLE__
	mov pc, lr
