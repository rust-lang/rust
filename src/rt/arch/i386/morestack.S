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
#define MORESTACK               ___morestack
#endif
#endif

.globl UPCALL_NEW_STACK
.globl UPCALL_DEL_STACK
#ifndef __APPLE__
.globl RUST_GET_TASK
.globl UPCALL_CALL_C_STACK
#endif
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

MORESTACK:
#if defined(__linux__) || defined(__APPLE__)
	.cfi_startproc
#endif

	pushl %ebp
#if defined(__linux__) || defined(__APPLE__)
	.cfi_def_cfa_offset 20
	.cfi_offset %ebp, -20
#endif
	movl %esp, %ebp
#if defined(__linux__) || defined(__APPLE__)
	.cfi_def_cfa_register %ebp
#endif

	// NB: This can be called with the fastcc convention so we
	// have to preserve any argument registers

	// NB: __morestack is called misaligned by 4 bytes, i.e.
	// subl $4, %esp would get us to a normal alignment

	subl $44,%esp

	// Save fastcc arguments
	movl %ecx, 28(%esp)
	movl %edx, 24(%esp)

	// FIXME (1226): main is compiled with the split-stack prologue,
	// causing it to call __morestack, so we have to jump back out
	calll RUST_GET_TASK
	testl %eax,%eax
	jz .L$bail

	// Save the the correct %esp value for our grandparent frame,
	// for the unwinder
	leal 20(%ebp), %eax
	movl %eax, -4(%ebp)

	// The arguments to rust_new_stack2
	movl 56(%esp),%eax  // Size of stack arguments
	movl %eax,20(%esp)
	leal 64(%esp),%eax // Address of stack arguments
	movl %eax,16(%esp)
	movl 52(%esp),%eax // The amount of stack needed
	movl %eax,12(%esp)
	movl $0, 8(%esp)   // Out pointer

#ifdef __APPLE__
	call 1f
1:	popl %eax
	movl L_upcall_new_stack$non_lazy_ptr-1b(%eax),%eax
	movl %eax, 4(%esp)
#else
	movl $UPCALL_NEW_STACK,4(%esp)
#endif

	leal 8(%esp), %eax
	movl %eax,(%esp)
	call UPCALL_CALL_C

	movl 48(%esp),%eax          // Grab the return pointer.
	inc  %eax    // Skip past the ret instruction in the parent fn

	// Restore fastcc arguments
	movl 28(%esp), %ecx
	movl 24(%esp), %edx

	movl 8(%esp),%esp          // Switch stacks.
	call *%eax                 // Re-enter the function that called us.

	// Now the function that called us has returned, so we need to delete the
	// old stack space.

	// Switch back to the rust stack
	movl %ebp, %esp

	// Remember that __morestack is called misaligned so %ebp
	// is not aligned to a 16-byte boundary, these 4 bytes realign.
	subl $4, %esp

	// Now that we're on the return path we want to avoid
	// stomping on %eax. FIXME: Need to save and restore
	// eax to actually preserve it across the call to delete the stack
#ifdef __APPLE__
	call 1f
1:	popl %ecx
	movl L_upcall_del_stack$non_lazy_ptr-1b(%ecx),%ecx
	pushl %ecx
#else
	pushl $UPCALL_DEL_STACK
#endif

	pushl $0
	call UPCALL_CALL_C

	addl $12,%esp

	popl %ebp
#if defined(__linux__) || defined(__APPLE__)
	.cfi_restore %ebp
	.cfi_def_cfa %esp, 16
#endif
	retl $8

.L$bail:
	movl 32(%esp),%eax
	inc %eax
	
	addl $44, %esp
	popl %ebp
	addl $4+8,%esp
	
	jmpl *%eax

#if defined(__linux__) || defined(__APPLE__)
	.cfi_endproc
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

