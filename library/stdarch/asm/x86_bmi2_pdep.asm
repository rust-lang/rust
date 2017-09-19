_pdep_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  pdepl	%esi, %edi, %eax
  popq	%rbp
  retq
_pdep_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  pdepq	%rsi, %rdi, %rax
  popq	%rbp
  retq
