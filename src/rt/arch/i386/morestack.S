    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a functiono
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define RUST_NEW_STACK      _rust_new_stack
#define RUST_DEL_STACK      _rust_del_stack
#define MORESTACK           ___morestack
#else
#define RUST_NEW_STACK      rust_new_stack
#define RUST_DEL_STACK      rust_del_stack
#define MORESTACK           __morestack
#endif

.globl RUST_NEW_STACK
.globl RUST_DEL_STACK

.globl MORESTACK

MORESTACK:
    pushl 8(%esp)           //               argsz > ra stksz argsz x x ra args
    leal 28(%esp),%eax      //                 argsz ra stksz argsz x x ra args 
    pushl %eax              //          argp > argsz ra stksz argsz x x ra args
    pushl 12(%esp)          //    stksz > argp argsz ra stksz argsz x x ra args
    calll RUST_NEW_STACK
    addl $12,%esp           //                       ra stksz argsz x x ra args

    movl (%esp),%edx        // Grab the return pointer.
    incl %edx               // Skip past the `ret`.
    movl %eax,%esp          // Switch to the new stack.
    calll *%edx             // Re-enter the function that called us.

    // Now the function that called us has returned, so we need to delete the
    // old stack space.
    calll RUST_DEL_STACK
    movl %eax,%esp
    retl $8                 //                       ra stksz argsz x x ra args

