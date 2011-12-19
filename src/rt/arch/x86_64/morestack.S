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

#if defined(__linux__)
	.hidden MORESTACK
#else
#if defined(__APPLE__)
	.private_extern MORESTACK
#endif
#endif

#ifdef __ELF__
	.type MORESTACK,@function
#endif


#if defined(__linux__) || defined(__APPLE__)
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

	pushq $0 // Alignment

	// FIXME: libgcc also saves rax. not sure if we need to

	// Save argument registers of the original function
	pushq	%rdi
	pushq	%rsi
	pushq	%rdx
	pushq	%rcx
	pushq	%r8
	pushq	%r9

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

	// Pop the saved arguments
	popq %r9
	popq %r8
	popq %rcx
	popq %rdx
	popq %rsi
	popq %rdi

	// Pop the unwinding %rsp
	addq $8, %rsp

        movq 8(%rbp),%r10       // Grab the return pointer.
        incq %r10               // Skip past the `ret` in our parent frame
        movq %rax,%rsp          // Switch to the new stack.

        call *%r10              // Reenter the caller function

	// Switch back to the rust stack
	movq %rbp, %rsp

	// Align the stack again
	pushq $0

	// FIXME: Should preserve %rax here
#ifdef __APPLE__
	call UPCALL_DEL_STACK
#endif
#ifdef __linux__
	call UPCALL_DEL_STACK@PLT
#endif

	addq $8, %rsp
	popq %rbp
	// FIXME: I don't think these rules are necessary
	// since the unwinder should never encounter an instruction
	// pointer pointing here.
	.cfi_restore %rbp
	.cfi_def_cfa %rsp, 16
	ret
	
	.cfi_endproc

#else
MORESTACK:
	ret
#endif
