    .text
    .global cc_plus_one_asm
    .type cc_plus_one_asm, @function
cc_plus_one_asm:
    movl (%rdi), %eax
    inc %eax
    retq
