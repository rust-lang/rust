_lzcnt_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  lzcntl	%edi, %eax
  popq	%rbp
  retq
_lzcnt_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  lzcntq	%rdi, %rax
  popq	%rbp
  retq
