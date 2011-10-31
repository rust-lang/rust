    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a functiono
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define RUST_NEW_STACK      _rust_new_stack
#define RUST_DEL_STACK      _rust_del_stack
#else
#define RUST_NEW_STACK      rust_new_stack
#define RUST_DEL_STACK      rust_del_stack
#endif

.globl RUST_NEW_STACK
.globl RUST_DEL_STACK

.globl __morestack

__morestack:
    pushl %edx              // param 2: size of arguments
    leal 8(%esp),%eax
    pushl %eax              // param 1: starting addr of arguments
    pushl %ecx              // param 0: amount of space needed
    calll RUST_NEW_STACK

    movl (%esp),%edx        // Grab the return pointer.
    incl %edx               // Skip past the `ret`.
    movl %eax,%esp          // Switch to the new stack.
    calll *%edx             // Enter the new function.

    // Now the function that called us has returned, so we need to delete the
    // old stack space.
    calll RUST_DEL_STACK
    movl %eax,%esp          // Switch back to the old stack.
    retl

