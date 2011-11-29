    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a function
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define RUST_NEW_STACK          _rust_new_stack
#define RUST_DEL_STACK          _rust_del_stack
#define RUST_GET_PREV_STACK     _rust_get_prev_stack
#define RUST_GET_TASK           _rust_get_task
#define UPCALL_ALLOC_C_STACK    _upcall_alloc_c_stack
#define UPCALL_CALL_C_STACK     _upcall_call_c_stack
#define MORESTACK               ___morestack
#else
#define RUST_NEW_STACK          rust_new_stack
#define RUST_DEL_STACK          rust_del_stack
#define RUST_GET_PREV_STACK     rust_get_prev_stack
#define RUST_GET_TASK           rust_get_task
#define UPCALL_ALLOC_C_STACK    upcall_alloc_c_stack
#define UPCALL_CALL_C_STACK     upcall_call_c_stack
#define MORESTACK               __morestack
#endif

#ifdef __APPLE__
#define ALIGNMENT               4
#else
#define ALIGNMENT               8
#endif

#if defined (__APPLE__)
#define NEW_STACK_ADDR          rust_new_stack_sym-.L$pic_ref_pt_0(%eax)
#define DEL_STACK_ADDR          rust_del_stack_sym-.L$pic_ref_pt_1(%edx)
#else
#if defined (_WIN32)
#define NEW_STACK_ADDR          $_rust_new_stack
#define DEL_STACK_ADDR          $_rust_del_stack
#else
#define NEW_STACK_ADDR          $rust_new_stack
#define DEL_STACK_ADDR          $rust_del_stack
#endif
#endif

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

    // Sanity check to make sure that there is a currently-running task.
    subl $12,%esp
    calll RUST_GET_TASK
    testl %eax,%eax
    jz .L$bail

    movl $12, (%esp)
    calll UPCALL_ALLOC_C_STACK
    movl %eax,%edx

    movl %esp, 12(%edx)
                            // C stack              | esp+12
                            // ---------------------+-------------------------
    movl 20(%esp),%eax      //                      | ra stksz argsz x ra args
    movl %eax,8(%edx)       //              argsz > | ra stksz argsz x ra args
    leal 28+ALIGNMENT(%esp),%eax //           argsz | ra stksz argsz x ra args
    movl %eax,4(%edx)       //         argp > argsz | ra stksz argsz x ra args
    movl 16(%esp),%eax      //           argp argsz | ra stksz argsz x ra args
    movl %eax,(%edx)        //   stksz > argp argsz | ra stksz argsz x ra args

    calll .L$pic_ref_pt_0
.L$pic_ref_pt_0:
    popl %eax

    movl NEW_STACK_ADDR,%eax
    movl %eax,(%esp)
    movl %edx,4(%esp)
    // FIXME: Don't understand why just _here_ I have to say @PLT
    calll UPCALL_CALL_C_STACK@PLT

    movl 12(%esp),%edx          // Grab the return pointer.
    addl $RETURN_OFFSET,%edx    // Skip past the `add esp,4` and the `ret`.

    movl %eax,%esp          // Switch stacks.
    calll *%edx             // Re-enter the function that called us.

    // Now the function that called us has returned, so we need to delete the
    // old stack space.

    calll RUST_GET_PREV_STACK
    movl %eax,%esp          // Switch back to the old stack.

    movl $0,(%esp)
    calll UPCALL_ALLOC_C_STACK

    calll .L$pic_ref_pt_1
.L$pic_ref_pt_1:
    popl %edx

    movl DEL_STACK_ADDR,%edx
    movl %edx,(%esp)
    movl %eax,4(%esp)
    calll UPCALL_CALL_C_STACK

    addl $12,%esp
    retl $8    //                      ra stksz argsz x ra args

.L$bail:
    movl 12(%esp),%edx
    addl $RETURN_OFFSET,%edx
    addl $12+4+8+ALIGNMENT,%esp
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

