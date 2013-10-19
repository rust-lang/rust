// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

/* See i386/morestack.S for the lengthy, general explanation. */

.text

#if defined(__APPLE__)
#define MORESTACK               ___morestack
#else
#define MORESTACK               __morestack
#endif

#if defined(__APPLE__)
#define EXHAUSTED               _rust_stack_exhausted
#elif defined(__linux__) || defined(__FreeBSD__)
#define EXHAUSTED               rust_stack_exhausted@PLT
#else
#define EXHAUSTED               rust_stack_exhausted
#endif

#if defined(__linux__) || defined(__FreeBSD__)
	.hidden MORESTACK
#else
#if defined(__APPLE__)
	.private_extern MORESTACK
#endif
#endif

#ifdef __ELF__
	.type MORESTACK,@function
#endif

.globl MORESTACK
MORESTACK:
	.cfi_startproc

	pushq %rbp
	// The CFA is 24 bytes above the register that it will
	// be associated with for this frame (%rbp). That is 8
	// bytes greater than a normal frame, to allow the unwinder
	// to skip the partial frame of the original function.
	.cfi_def_cfa_offset 24
	// %rbp is -24 bytes from the CFA
	.cfi_offset %rbp, -24
	movq %rsp, %rbp
	// Calculate the CFA as on offset from %ebp
	.cfi_def_cfa_register %rbp

        // re-align the stack
        subq $8, %rsp

        // kill this program
        call EXHAUSTED

        // the exhaustion function guarantees that it can't return

	.cfi_endproc
