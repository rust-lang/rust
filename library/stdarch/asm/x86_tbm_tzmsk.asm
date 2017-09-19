_tzmsk_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  tzmsk	%edi, %eax
  popq	%rbp
  retq
_tzmsk_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  tzmsk	%rdi, %rax
  popq	%rbp
  retq
