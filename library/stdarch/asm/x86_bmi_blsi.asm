_blsi_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blsil	%edi, %eax
  popq	%rbp
  retq
_blsi_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blsiq	%rdi, %rax
  popq	%rbp
  retq
