_pext_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  pextl	%esi, %edi, %eax
  popq	%rbp
  retq
_pext_u64:
	pushq	%rbp
	movq	%rsp, %rbp
	pextq	%rsi, %rdi, %rax
	popq	%rbp
	retq
