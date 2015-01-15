// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section       .note.GNU-stack, "", %progbits
#endif

/* See i386/morestack.S for the lengthy, general explanation. */

.global rust_stack_exhausted

.hidden __morestack

// FIXME(POWERPC): this might not be perfectly right but works for now
__morestack:
       .cfi_startproc
       bl rust_stack_exhausted
       // the above function ensures that it never returns
       .cfi_endproc
.end  __morestack
