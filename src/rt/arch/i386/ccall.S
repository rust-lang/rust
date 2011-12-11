    .text

#if defined(__APPLE__) || defined(_WIN32)
.globl _asm_call_on_stack
_asm_call_on_stack:
#else
.globl asm_call_on_stack
asm_call_on_stack:
#endif

#if defined(__linux__) || defined(__APPLE__)
	.cfi_startproc
#endif

	pushl %ebp

#if defined(__linux__) || defined(__APPLE__)
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
#endif

	movl %esp,%ebp          // save esp

#if defined(__linux__) || defined(__APPLE__)
	.cfi_def_cfa_register %ebp
#endif

	movl 16(%ebp),%esp      // load new esp
	subl $12,%esp           // maintain 16-byte alignment
	pushl 8(%ebp)           // push ptr to argument block
	calll *12(%ebp)
	movl %ebp,%esp          // would like to use "leave" but it's slower
	popl %ebp

	ret

#if defined(__linux__) || defined(__APPLE__)
	.cfi_endproc
#endif