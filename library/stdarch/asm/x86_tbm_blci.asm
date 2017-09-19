_blci_u32:
  pushq	%rbp
  movq	%rsp, %rbp
  blci	%edi, %eax
  popq	%rbp
  retq
_blci_u64:
  pushq	%rbp
  movq	%rsp, %rbp
  blci	%rdi, %rax
  popq	%rbp
  retq
