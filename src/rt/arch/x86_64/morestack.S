    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a functiono
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define RUST_NEW_STACK2     _rust_new_stack2
#define RUST_DEL_STACK      _rust_del_stack
#define UPCALL_CALL_C       _upcall_call_shim_on_c_stack
#define MORESTACK           ___morestack
#else
#define RUST_NEW_STACK2     rust_new_stack2
#define RUST_DEL_STACK      rust_del_stack
#define UPCALL_CALL_C       upcall_call_shim_on_c_stack
#define MORESTACK           __morestack
#endif

        // Naturally, nobody can agree as to
        // which arguments should go in which
        // registers:
#if defined(_WIN32)
#  define ARG0 %rcx
#  define ARG1 %rdx
#  define ARG2 %r8
#else
#  define ARG0 %rdi
#  define ARG1 %rsi
#  define ARG2 %rdx
#endif

.globl RUST_NEW_STACK2
.globl RUST_DEL_STACK
.globl UPCALL_CALL_C
.globl MORESTACK

#ifdef __ELF__
	.type MORESTACK,@function
#endif

MORESTACK:
	.cfi_startproc

	# Set up a normal backtrace
	pushq %rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq %rsp, %rbp
	.cfi_def_cfa_register %rbp

	// Save argument registers
	pushq	%rdi
	pushq	%rsi
	pushq	%rdx
	pushq	%rcx
	pushq	%r8
	pushq	%r9

	// Calculate the address of the stack arguments
	movq %rbp, %rcx
	addq $16, %rcx   // Add the saved %rbp, and return address
	addq %r11, %rcx // Add the size of stack arguments

	pushq %r10 // The amount of stack needed
	pushq %rcx // Address of stack arguments
	pushq %r11 // Size of stack arguments
	pushq %rbp // Save the Rust stack pointer

	// FIXME: Don't understand why I have to use the PLT here
	lea RUST_NEW_STACK2@PLT(%rip), %rsi
	lea 24(%rsp), %rdi
	call UPCALL_CALL_C@PLT
	
        mov (%rsp),%rdx        // Grab the return pointer.
        inc %rdx               // Skip past the `ret`.
        mov %rax,%rsp          // Switch to the new stack.
        call *%rdx             // Enter the new function.

	.cfi_endproc