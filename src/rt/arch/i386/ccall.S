    .text

// upcall_call_c_stack(void (*fn)(), void *new_esp)
.globl _upcall_call_c_stack
_upcall_call_c_stack:
    movl %esp,%ecx          // grab esp
    movl 8(%esp),%eax       // save fn
    movl 12(%esp),%esp      // switch stack
    pushl %ecx              // save esp on stack
    calll *%eax
    popl %esp               // restore esp
    ret


