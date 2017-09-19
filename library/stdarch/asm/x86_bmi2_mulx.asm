_umulx_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  movl	%edi, %ecx
  movl	%esi, %eax
  imulq	%rcx, %rax
  popq	%rbp
  retq
_umulx_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  mulxq	%rsi, %rcx, %rax
  movq	%rcx, (%rdi)
  movq	%rax, 8(%rdi)
  movq	%rdi, %rax
  popq	%rbp
  retq
