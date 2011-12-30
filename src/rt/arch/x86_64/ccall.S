/*
	The function for switching to the C stack.  It is called
	__morestack because gdb allows any frame with that name to
	move the stack pointer to a different stack, which it usually
	considers an error.
*/

#include "regs.h"

#define ARG0 RUSTRT_ARG0_S
#define ARG1 RUSTRT_ARG1_S
#define ARG2 RUSTRT_ARG2_S
        
        .text

#if defined(__APPLE__) || defined(_WIN32)
.globl ___morestack
.private_extern MORESTACK
___morestack:
#else
.globl __morestack
.hidden __morestack
__morestack:
#endif

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
	.cfi_startproc
#endif

	push %rbp
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
#endif

	mov %rsp,%rbp          // save rsp

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
	.cfi_def_cfa_register %rbp
#endif

	mov ARG2,%rsp          // switch stack
	call *ARG1             // invoke target address
	mov %rbp,%rsp
	pop %rbp

	ret

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
	.cfi_endproc
#endif
