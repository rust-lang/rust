    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a functiono
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define UPCALL_NEW_STACK        _upcall_new_stack
#define UPCALL_DEL_STACK        _upcall_del_stack
#define UPCALL_CALL_C           _upcall_call_shim_on_c_stack
#define MORESTACK               ___morestack
#else
#define UPCALL_NEW_STACK        upcall_new_stack
#define UPCALL_DEL_STACK        upcall_del_stack
#define UPCALL_CALL_C           upcall_call_shim_on_c_stack
#define MORESTACK               __morestack
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

.globl UPCALL_NEW_STACK
.globl UPCALL_DEL_STACK
.globl UPCALL_CALL_C
.globl MORESTACK

// FIXME: What about _WIN32?	
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
	
	// Set up a normal backtrace
	pushq %rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq %rsp, %rbp
	.cfi_def_cfa_register %rbp

	// During unwinding we want to skip our caller since it's not
	// a complete frame and will make the unwinder sad
	// Don't understand this line
	.cfi_offset 16, 0
	// Tell the unwinding where to get the stack pointer for
	// our grandparent frame
	.cfi_offset %rsp, -24

	// Save the grandparent stack pointer for the unwinder
	leaq 16(%rbp), %rax
	pushq %rax

	// During unwinding we want to skip our caller since it's not
	// a complete frame and will make the unwinder sad
	// Don't understand this line
	.cfi_offset 16, 0
	// Tell the unwinding where to get the stack pointer for
	// our grandparent frame
	.cfi_offset %rsp, -24

	// Save the grandparent stack pointer for the unwinder
	leaq 16(%rbp), %rax
	pushq %rax

	// FIXME: libgcc also saves rax. not sure if we need to

	// Save argument registers
	pushq	%rdi
	pushq	%rsi
	pushq	%rdx
	pushq	%rcx
	pushq	%r8
	pushq	%r9

	// Calculate the address of the stack arguments.
	// We have the base pointer, __morestack's return address,
	// and __morestack's caller's return address to skip
	movq %rbp, %rcx
	addq $24, %rcx  // Base pointer, return address x2

	pushq %r11 // Size of stack arguments
	pushq %rcx // Address of stack arguments
	pushq %r10 // The amount of stack needed
	pushq $0   // Out pointer

	movq UPCALL_NEW_STACK@GOTPCREL(%rip), %rsi
	movq %rsp, %rdi
#ifdef __APPLE__
	call UPCALL_CALL_C
#endif
#ifdef __linux__
	call UPCALL_CALL_C@PLT
#endif

	// Pop the new_stack_args struct
	popq %rax
	addq $24, %rsp

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
	
	movq UPCALL_DEL_STACK@GOTPCREL(%rip), %rsi
	movq $0, %rdi
#ifdef __APPLE__
	call UPCALL_CALL_C
#endif
#ifdef __linux__
	call UPCALL_CALL_C@PLT
#endif

	addq $8, %rsp
	popq %rbp
	.cfi_restore %rbp
	.cfi_def_cfa %rsp, 8
	ret
	
	.cfi_endproc

#else
MORESTACK:
	ret
#endif
