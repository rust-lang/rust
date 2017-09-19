_bzhi_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  bzhil	%esi, %edi, %eax
  popq	%rbp
  retq
_bzhi_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  bzhiq	%rsi, %rdi, %rax
  popq	%rbp
  retq
