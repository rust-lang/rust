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
    mov %rsi,%rsp          // switch stack
    call *%rdi
    mov %rbp,%rsp          // would like to use "leave" but it's slower
    pop %rbp
    ret

