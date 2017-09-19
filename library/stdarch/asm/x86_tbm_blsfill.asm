_blsfill_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blsfill	%edi, %eax
  popq	%rbp
  retq
_blsfill_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blsfill	%rdi, %rax
  popq	%rbp
  retq
