    .text
    .global __memcpy_wordaligned
    .type   __memcpy_wordaligned,@function
__memcpy_wordaligned:
    /* r6 is dest, r7 is src, r8 is count. */
    /* r8 is assumed to be both nonzero and a multiple of 4. */
    add -4, r3
    st.w r6, 0[r3]
    .align 4 /* Branch targets should be on a word boundary. May insert a mov r0, r0. */
.Lmemcpy_loop:
    /** Looks out of order, but avoid some hazards indexing from a just-updated register */
    add 4, r7
    add 4, r6

    /** Update the branch condition well before using it */
    add -4, r8

    /** ld should precede st per V810 slides */
    /** Use -4 offset since we already incremented it */
    ld.w -4[r7], r10
    st.w r10, -4[r6]

    bnz .Lmemcpy_loop
    ld.w 0[r3], r10
    add 4, r3
    jmp [r31]
.Lmemcpy_end:
    .size	__memcpy_wordaligned, .Lmemcpy_end-__memcpy_wordaligned
