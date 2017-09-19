_blcfill_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blcfill	%edi, %eax
  popq	%rbp
  retq
_blcfill_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blcfill	%rdi, %rax
  popq	%rbp
  retq
