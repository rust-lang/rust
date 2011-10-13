    .text

// upcall_call_c_stack(void (*fn)(), void *new_esp)
//
// Note that we could use |enter| and |leave| but the manuals tell me they're
// slower.
#if defined(__APPLE__) || defined(_WIN32)
.globl _upcall_call_c_stack
_upcall_call_c_stack:
#else
.globl upcall_call_c_stack
upcall_call_c_stack:
#endif
    pushl %rbp
    movl %rsp,%rbp          // save rsp
    movl %rsi,%rsp          // switch stack
    calll *%rdi
    movl %rbp,%rsp          // would like to use "leave" but it's slower
    popl %rbp
    ret

