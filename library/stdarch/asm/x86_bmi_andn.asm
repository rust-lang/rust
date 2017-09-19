_andn_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  andnl	%esi, %edi, %eax
  popq	%rbp
  retq
_andn_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  andnq	%rsi, %rdi, %rax
  popq	%rbp
  retq
