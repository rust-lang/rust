_bextr_u32:
	pushq	%rbp
	movq	%rsp, %rbp
	movzbl	%sil, %eax
	shll	$8, %edx
	movzwl	%dx, %ecx
	orl	%eax, %ecx
	bextrl	%ecx, %edi, %eax
	popq	%rbp
	retq
_bextr_u64:
	pushq	%rbp
	movq	%rsp, %rbp
	movzbl	%sil, %eax
	shlq	$8, %rdx
	movzwl	%dx, %ecx
	orq	%rax, %rcx
	bextrq	%rcx, %rdi, %rax
	popq	%rbp
	retq
_bextr2_u32:
	pushq	%rbp
	movq	%rsp, %rbp
	bextrl	%esi, %edi, %eax
	popq	%rbp
	retq
_bextr2_u64:
	pushq	%rbp
	movq	%rsp, %rbp
	bextrq	%rsi, %rdi, %rax
	popq	%rbp
	retq
