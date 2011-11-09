#include "regs.h"

#define ARG0 RUSTRT_ARG0_S
#define ARG1 RUSTRT_ARG1_S
        
        .text

// upcall_call_c_stack(void (*fn)(), void *new_esp)
//
// Note that we could use |enter| and |leave| but the manuals tell me they're
// slower.
#if defined(__APPLE__) || defined(_WIN32)
.globl _upcall_call_c_stack
.globl _upcall_call_c_stack_i64
.globl _upcall_call_c_stack_float
_upcall_call_c_stack:
_upcall_call_c_stack_i64:
_upcall_call_c_stack_float:
#else
.globl upcall_call_c_stack
.globl upcall_call_c_stack_i64
.globl upcall_call_c_stack_float
upcall_call_c_stack:
upcall_call_c_stack_i64:
upcall_call_c_stack_float:
#endif
    push %rbp
    mov %rsp,%rbp          // save rsp
    mov ARG1,%rsp          // switch stack

    // Hack: the arguments to the function are sitting
    // on the stack right now, as in i386 calling
    // convention.  We need them in registers.
    // For now, we just load them into registers.
    //
    // This is a total hack because it does not consider
    // the actual arguments of the target function.
    // It fails if there are non-INTEGER class arguments,
    // which would get pushed on the stack, or if there are
    // additional arguments beyond those that will get
    // passed in registers.
    mov ARG0,%r11     // Remember target address
    mov  0(%rsp),RUSTRT_ARG0_S
    mov  8(%rsp),RUSTRT_ARG1_S
    mov 16(%rsp),RUSTRT_ARG2_S
    mov 24(%rsp),RUSTRT_ARG3_S
#   ifdef RUSTRT_ARG4_S        
    mov 32(%rsp),RUSTRT_ARG4_S
#   endif
#   ifdef RUSTRT_ARG5_S        
    mov 40(%rsp),RUSTRT_ARG5_S
#   endif
        
    call *%r11
    mov %rbp,%rsp          // would like to use "leave" but it's slower
    pop %rbp
    ret

