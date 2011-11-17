    .text

// __morestack
//
// LLVM generates a call to this to allocate more stack space in a functiono
// prolog when we run out.

#if defined(__APPLE__) || defined(_WIN32)
#define RUST_NEW_STACK      _rust_new_stack
#define RUST_DEL_STACK      _rust_del_stack
#define MORESTACK           __morestack
#else
#define RUST_NEW_STACK      rust_new_stack
#define RUST_DEL_STACK      rust_del_stack
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

.globl RUST_NEW_STACK
.globl RUST_DEL_STACK

.globl MORESTACK

MORESTACK:
        // Hastily and probably incorrectly ported from i386 version.
        // Actually this calling convention doens't make so much sense
        // for x86_64...
        mov %rcx, ARG0      // param 0: amount of space needed
        mov %rdx, ARG2      // param 2: size of arguments
        lea 8(%rsp),ARG1
        call rust_new_stack_sym

        mov (%rsp),%rdx        // Grab the return pointer.
        inc %rdx               // Skip past the `ret`.
        mov %rax,%rsp          // Switch to the new stack.
        call *%rdx             // Enter the new function.

        // Now the function that called us has returned, so we need to delete the
        // old stack space.
        call rust_new_stack_sym
        mov %rax,%rsp          // Switch back to the old stack.
        ret

// This is totally broken
rust_new_stack_sym:
rust_del_stack_sym: