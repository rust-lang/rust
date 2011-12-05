    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a function
// prolog when we run out.

#if defined(__APPLE__)
#define RUST_GET_TASK           L_rust_get_task$stub
#define UPCALL_CALL_C           L_upcall_call_shim_on_c_stack$stub
#define MORESTACK               ___morestack
#else
#if defined(__linux__)
#define UPCALL_NEW_STACK        upcall_new_stack
#define UPCALL_DEL_STACK        upcall_del_stack
#define RUST_GET_TASK           rust_get_task
#define UPCALL_CALL_C           upcall_call_shim_on_c_stack
#define MORESTACK               __morestack
#else
#define UPCALL_NEW_STACK        _upcall_new_stack
#define UPCALL_DEL_STACK        _upcall_del_stack
#define RUST_GET_TASK           _rust_get_task
#define UPCALL_CALL_C           _upcall_call_shim_on_c_stack
#define MORESTACK               __morestack
#endif
#endif

.globl UPCALL_NEW_STACK
.globl UPCALL_DEL_STACK
.globl RUST_GET_TASK
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

#if defined(__linux__) || defined(__APPLE__)
MORESTACK:

	pushl %ebp
	movl %esp, %ebp
	
	// FIXME (1226): main is compiled with the split-stack prologue,
	// causing it to call __morestack, so we have to jump back out
	subl $28,%esp
	calll RUST_GET_TASK
	testl %eax,%eax
	jz .L$bail

	// The arguments to rust_new_stack2
	movl 40(%esp),%eax  // Size of stack arguments
	movl %eax,16(%esp)
	leal 48(%esp),%eax // Address of stack arguments
	movl %eax,12(%esp)
	movl 36(%esp),%eax // The amount of stack needed
	movl %eax,8(%esp)

#ifdef __APPLE__
	call 1f
1:	popl %ebx
	movl L_upcall_new_stack$non_lazy_ptr-1b(%ebx),%ecx
	movl %ecx, 4(%esp)
#else
	movl $UPCALL_NEW_STACK,4(%esp)
#endif

	leal 8(%esp), %eax
	movl %eax,(%esp)
	call UPCALL_CALL_C

	movl 32(%esp),%edx          // Grab the return pointer.
	inc  %edx    // Skip past the ret instruction in the parent fn

	movl %eax,%esp          // Switch stacks.
	call *%edx             // Re-enter the function that called us.

	// Now the function that called us has returned, so we need to delete the
	// old stack space.

	// Switch back to the rust stack
	movl %ebp, %esp

	subl $8, %esp // Alignment

#ifdef __APPLE__
	call 1f
1:	popl %ebx
	movl L_upcall_del_stack$non_lazy_ptr-1b(%ebx),%ecx
	pushl %ecx
#else
	pushl $UPCALL_DEL_STACK
#endif

	pushl $0
	call UPCALL_CALL_C

	addl $16,%esp
	popl %ebp
	retl $8

.L$bail:
	movl 32(%esp),%edx
	inc %edx
	
	addl $28, %esp
	popl %ebp
	addl $4+8,%esp
	
	jmpl *%edx


#else
MORESTACK:
	ret
#endif

#ifdef __APPLE__

	.section __IMPORT,__pointers,non_lazy_symbol_pointers
L_upcall_new_stack$non_lazy_ptr:
	.indirect_symbol _upcall_new_stack
	.long 0
L_upcall_del_stack$non_lazy_ptr:
	.indirect_symbol _upcall_del_stack
	.long 0

.section __IMPORT,__jump_table,symbol_stubs,pure_instructions+self_modifying_code,5

	// Linker will replace the hlts (the ascii) with jmp
L_rust_get_task$stub:
	.indirect_symbol _rust_get_task
	.ascii	 "\364\364\364\364\364"

L_upcall_call_shim_on_c_stack$stub:
	.indirect_symbol _upcall_call_shim_on_c_stack
	.ascii	 "\364\364\364\364\364"

	.subsections_via_symbols
#endif

