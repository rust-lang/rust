_blcic_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blcic	%edi, %eax
  popq	%rbp
  retq
_blcic_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blcic	%rdi, %rax
  popq	%rbp
  retq
