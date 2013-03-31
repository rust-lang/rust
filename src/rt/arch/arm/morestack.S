// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

.text
.code 32
.arm
.align

.globl __morestack
.hidden __morestack
__morestack:
