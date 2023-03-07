    .text
    .global cmake_plus_one_asm
    .type cmake_plus_one_asm, @function
cmake_plus_one_asm:
    movl (%rdi), %eax
    inc %eax
    retq
