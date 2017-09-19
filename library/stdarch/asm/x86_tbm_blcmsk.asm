_blcmsk_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blcmsk	%edi, %eax
  popq	%rbp
  retq
_blcmsk_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blcmsk	%rdi, %rax
  popq	%rbp
  retq
