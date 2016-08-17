use core::intrinsics;

// NOTE These functions are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function
#[cfg(windows)]
#[naked]
#[cfg_attr(not(test), no_mangle)]
pub unsafe fn ___chkstk_ms() {
    asm!("push   %rcx
          push   %rax
          cmp    $$0x1000,%rax
          lea    24(%rsp),%rcx
          jb     1f
          2:
          sub    $$0x1000,%rcx
          test   %rcx,(%rcx)
          sub    $$0x1000,%rax
          cmp    $$0x1000,%rax
          ja     2b
          1:
          sub    %rax,%rcx
          test   %rcx,(%rcx)
          pop    %rax
          pop    %rcx
          ret");
    intrinsics::unreachable();
}

#[cfg(windows)]
#[naked]
#[cfg_attr(not(test), no_mangle)]
pub unsafe fn __alloca() {
    asm!("mov    %rcx,%rax  // x64 _alloca is a normal function with parameter in rcx
          jmp    ___chkstk  // Jump to ___chkstk since fallthrough may be unreliable");
    intrinsics::unreachable();
}

#[cfg(windows)]
#[naked]
#[cfg_attr(not(test), no_mangle)]
pub unsafe fn ___chkstk() {
    asm!("push   %rcx
          cmp    $$0x1000,%rax
          lea    16(%rsp),%rcx  // rsp before calling this routine -> rcx
          jb     1f
          2:
          sub    $$0x1000,%rcx
          test   %rcx,(%rcx)
          sub    $$0x1000,%rax
          cmp    $$0x1000,%rax
          ja     2b
          1:
          sub    %rax,%rcx
          test   %rcx,(%rcx)

          lea    8(%rsp),%rax   // load pointer to the return address into rax
          mov    %rcx,%rsp      // install the new top of stack pointer into rsp
          mov    -8(%rax),%rcx  // restore rcx
          push   (%rax)         // push return address onto the stack
          sub    %rsp,%rax      // restore the original value in rax
          ret");
    intrinsics::unreachable();
}

