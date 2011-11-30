    .text

#if defined(__APPLE__) || defined(_WIN32)
.globl _asm_call_on_stack
_asm_call_on_stack:
#else
.globl asm_call_on_stack
asm_call_on_stack:
#endif
    pushl %ebp
    movl %esp,%ebp          // save esp
    movl 16(%ebp),%esp      // load new esp
    subl $12,%esp           // maintain 16-byte alignment
    pushl 8(%ebp)           // push ptr to argument block
    calll *12(%ebp)
    movl %ebp,%esp          // would like to use "leave" but it's slower
    popl %ebp
    ret
