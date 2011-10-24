    .text

// upcall_call_c_stack(void (*fn)(), void *new_esp)
//
// Note that we could use |enter| and |leave| but the manuals tell me they're
// slower.
#if defined(__APPLE__) || defined(_WIN32)
.globl _upcall_call_c_stack
.globl _upcall_call_c_stack_float
_upcall_call_c_stack:
_upcall_call_c_stack_float:
#else
.globl upcall_call_c_stack
.globl upcall_call_c_stack_float
upcall_call_c_stack:
upcall_call_c_stack_float:
#endif
    pushl %ebp
    movl %esp,%ebp          // save esp
    movl 8(%esp),%eax       // eax = callee
    movl 12(%esp),%esp      // switch stack
    calll *%eax
    movl %ebp,%esp          // would like to use "leave" but it's slower
    popl %ebp
    ret

