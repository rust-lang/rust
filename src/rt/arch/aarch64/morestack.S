#include "macros.S"

// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

/* See i386/morestack.S for the lengthy, general explanation. */

#if defined(__APPLE__)
#define MORESTACK ___morestack
#define STACK_EXHAUSTED _rust_stack_exhausted@plt
#else
#define MORESTACK __morestack
#define STACK_EXHAUSTED rust_stack_exhausted
#endif

.global STACK_EXHAUSTED

#if defined(__APPLE__)
.private_extern MORESTACK
#else
.hidden MORESTACK
#endif

#if !defined(__APPLE__)
func MORESTACK
#endif

// FIXME(AARCH64): this might not be perfectly right but works for now
MORESTACK:
	.cfi_startproc
	bl STACK_EXHAUSTED
	// the above function ensures that it never returns
	.cfi_endproc

#if !defined(__APPLE__)
endfunc MORESTACK
#endif
