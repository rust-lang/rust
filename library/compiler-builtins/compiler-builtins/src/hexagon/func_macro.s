 .macro FUNCTION_BEGIN name
 .text
 .p2align 5
 .globl \name
 .type \name, @function
\name:
 .endm

 .macro FUNCTION_END name
 .size \name, . - \name
 .endm

