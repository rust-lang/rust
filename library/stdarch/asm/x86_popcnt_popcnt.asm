_popcnt_u32:
	pushq	%rbp
	movq	%rsp, %rbp
	popcntl	%edi, %eax
	popq	%rbp
	retq
_popcnt_u64:
	pushq	%rbp
	movq	%rsp, %rbp
	popcntq	%rdi, %rax
	popq	%rbp
	retq
