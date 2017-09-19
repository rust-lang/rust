_blsic_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blsic	%edi, %eax
  popq	%rbp
  retq
_blsic_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blsic	%rdi, %rax
  popq	%rbp
  retq
