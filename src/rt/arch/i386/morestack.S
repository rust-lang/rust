    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a function
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define RUST_NEW_STACK2         _rust_new_stack2
#define RUST_DEL_STACK          _rust_del_stack
#define RUST_GET_PREV_STACK     _rust_get_prev_stack
#define RUST_GET_TASK           _rust_get_task
#define UPCALL_CALL_C           _upcall_call_shim_on_c_stack
#define MORESTACK               ___morestack
#else
#define RUST_NEW_STACK2         rust_new_stack2
#define RUST_DEL_STACK          rust_del_stack
#define RUST_GET_PREV_STACK     rust_get_prev_stack
#define RUST_GET_TASK           rust_get_task
#define UPCALL_CALL_C           upcall_call_shim_on_c_stack
#define MORESTACK               __morestack
#endif

#define ALIGNMENT 8
#define RETURN_OFFSET   7

.globl RUST_NEW_STACK
.globl RUST_DEL_STACK
.globl RUST_GET_PREV_STACK
.globl RUST_GET_TASK
.globl UPCALL_ALLOC_C_STACK
.globl UPCALL_CALL_C_STACK
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

#if defined(__linux__)
MORESTACK:

	pushl %ebp
	movl %esp, %ebp
	
	// FIXME (1226): main is compiled with the split-stack prologue,
	// causing it to call __morestack, so we have to jump back out
	subl $24,%esp
	calll RUST_GET_TASK
	testl %eax,%eax
	jz .L$bail

	// The arguments to rust_new_stack2
	movl %esp, 20(%esp) // Save the stack pointer
	movl 36(%esp),%eax  // Size of stack arguments
	movl %eax,16(%esp)
	leal 44+ALIGNMENT(%esp),%eax // Address of stack arguments
	movl %eax,12(%esp)
	movl 32(%esp),%eax // The amount of stack needed
	movl %eax,8(%esp)

	movl $RUST_NEW_STACK2,4(%esp)
	leal 8(%esp), %eax
	movl %eax,(%esp)
	call UPCALL_CALL_C

	movl 28(%esp),%edx          // Grab the return pointer.
	addl $RETURN_OFFSET,%edx    // Skip past the `add esp,4` and the `ret`.

	movl %eax,%esp          // Switch stacks.
	call *%edx             // Re-enter the function that called us.

	// Now the function that called us has returned, so we need to delete the
	// old stack space.

	// NB: This is assuming we already have at least 2 words
	// pushed onto the C stack. This is always true because
	// Rust functions have implicit arguments.
	movl $RUST_GET_PREV_STACK,4(%esp)
	movl $0, (%esp)
	call UPCALL_CALL_C

	// Switch back to the rust stack
	movl %eax, %esp

	movl $RUST_DEL_STACK,4(%esp)
	movl $0, (%esp)
	call UPCALL_CALL_C

	addl $24,%esp
	popl %ebp
	retl $8

.L$bail:
	movl 28(%esp),%edx
	addl $RETURN_OFFSET,%edx
	
	addl $24, %esp
	popl %ebp
	addl $4+8+ALIGNMENT,%esp
	
	jmpl *%edx


#else
MORESTACK:
	ret
#endif

#ifdef __APPLE__

		.section	__IMPORT,__pointers,non_lazy_symbol_pointers
rust_new_stack_sym:
    .indirect_symbol RUST_NEW_STACK
    .long 0
rust_del_stack_sym:
    .indirect_symbol RUST_DEL_STACK
    .long 0

#endif

