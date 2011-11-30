#include "regs.h"

#define ARG0 RUSTRT_ARG0_S
#define ARG1 RUSTRT_ARG1_S
#define ARG2 RUSTRT_ARG2_S
        
        .text

#if defined(__APPLE__) || defined(_WIN32)
.globl _asm_call_on_stack
_asm_call_on_stack:
#else
.globl asm_call_on_stack
asm_call_on_stack:
#endif
    push %rbp
    mov %rsp,%rbp          // save rsp
    mov ARG2,%rsp          // switch stack
    call *ARG1             // invoke target address
    mov %rbp,%rsp
    pop %rbp
    ret
