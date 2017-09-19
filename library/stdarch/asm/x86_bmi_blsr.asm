_blsr_u32:
	pushq	%rbp
	movq	%rsp, %rbp
	blsrl	%edi, %eax
	popq	%rbp
	retq
_blsr_u64:
	pushq	%rbp
	movq	%rsp, %rbp
	blsrq	%rdi, %rax
	popq	%rbp
	retq
