_blcs_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blcs	%edi, %eax
  popq	%rbp
  retq
_blcs_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blcs	%rdi, %rax
  popq	%rbp
  retq
