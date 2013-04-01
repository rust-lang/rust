// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

/*
	__morestack

	See i386/morestack.S for the lengthy, general explanation.
*/

.text

#if defined(__APPLE__) || defined(_WIN32)
#define UPCALL_NEW_STACK        _upcall_new_stack
#define UPCALL_DEL_STACK        _upcall_del_stack
#define MORESTACK               ___morestack
#else
#define UPCALL_NEW_STACK        upcall_new_stack
#define UPCALL_DEL_STACK        upcall_del_stack
#define MORESTACK               __morestack
#endif

.globl UPCALL_NEW_STACK
.globl UPCALL_DEL_STACK
.globl MORESTACK

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


#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
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

        subq $184, %rsp

	// Save argument registers of the original function
	movq %rdi,       (%rsp)
	movq %rsi,      8(%rsp)
	movq %rdx,     16(%rsp)
	movq %rcx,     24(%rsp)
	movq %r8,      32(%rsp)
	movq %r9,      40(%rsp)
	movdqa %xmm0,  48(%rsp)
	movdqa %xmm1,  64(%rsp)
	movdqa %xmm2,  80(%rsp)
	movdqa %xmm3,  96(%rsp)
	movdqa %xmm4, 112(%rsp)
	movdqa %xmm5, 128(%rsp)
	movdqa %xmm6, 144(%rsp)
	movdqa %xmm7, 160(%rsp)

	// Calculate the address of the stack arguments.
	// We have the base pointer, __morestack's return address,
	// and __morestack's caller's return address to skip
	movq %rbp, %rax
	addq $24, %rax  // Base pointer, return address x2

	// The arguments to __morestack are passed in %r10 & %r11

	movq %r11, %rdx // Size of stack arguments
	movq %rax, %rsi // Address of stack arguments
	movq %r10, %rdi // The amount of stack needed

#ifdef __APPLE__
	call UPCALL_NEW_STACK
#endif
#ifdef __linux__
	call UPCALL_NEW_STACK@PLT
#endif
#ifdef __FreeBSD__
	call UPCALL_NEW_STACK@PLT
#endif

	// Pop the saved arguments
	movq      (%rsp), %rdi
	movq     8(%rsp), %rsi
	movq    16(%rsp), %rdx
	movq    24(%rsp), %rcx
	movq    32(%rsp), %r8
	movq    40(%rsp), %r9
	movdqa  48(%rsp), %xmm0
	movdqa  64(%rsp), %xmm1
	movdqa  80(%rsp), %xmm2
	movdqa  96(%rsp), %xmm3
	movdqa 112(%rsp), %xmm4
	movdqa 128(%rsp), %xmm5
	movdqa 144(%rsp), %xmm6
	movdqa 160(%rsp), %xmm7

	addq $184, %rsp

        movq 8(%rbp),%r10       // Grab the return pointer.
        incq %r10               // Skip past the `ret` in our parent frame
        movq %rax,%rsp          // Switch to the new stack.

        call *%r10              // Reenter the caller function

	// Switch back to the rust stack
	movq %rbp, %rsp

	// Save the return value
	pushq %rax

#ifdef __APPLE__
	call UPCALL_DEL_STACK
#endif
#ifdef __linux__
	call UPCALL_DEL_STACK@PLT
#endif
#ifdef __FreeBSD__
	call UPCALL_DEL_STACK@PLT
#endif

	popq %rax // Restore the return value
	popq %rbp
	ret

	.cfi_endproc

#else
MORESTACK:
	ret
#endif
