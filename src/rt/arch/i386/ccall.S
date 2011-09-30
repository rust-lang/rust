    .text

// upcall_call_c_stack(void (*fn)(), void *new_esp)
//
// Note that we could use |enter| and |leave| but the manuals tell me they're
// slower.
.globl _upcall_call_c_stack
_upcall_call_c_stack:
    pushl %ebp
    movl %esp,%ebp          // save esp
    movl 8(%esp),%eax       // eax = callee
    movl 12(%esp),%esp      // switch stack
    calll *%eax
    movl %ebp,%esp          // would like to use "leave" but it's slower
    popl %ebp
    ret

