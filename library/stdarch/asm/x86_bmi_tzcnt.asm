_tzcnt_u32:
	pushq	%rbp
	movq	%rsp, %rbp
	tzcntl	%edi, %eax
	popq	%rbp
	retq
_tzcnt_u64:
	pushq	%rbp
	movq	%rsp, %rbp
	tzcntq	%rdi, %rax
	popq	%rbp
	retq
