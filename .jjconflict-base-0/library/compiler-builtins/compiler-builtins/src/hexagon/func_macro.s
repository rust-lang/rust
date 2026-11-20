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

 .macro FALLTHROUGH_TAIL_CALL name0 name1
 .size \name0, . - \name0
 .globl \name1
 .type \name1, @function
 .falign
\name1:
 .endm

