_t1mskc_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  t1mskc	%edi, %eax
  popq	%rbp
  retq
_t1mskc_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  t1mskc	%rdi, %rax
  popq	%rbp
  retq
