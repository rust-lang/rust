#![allow(unused_imports)]

use core::intrinsics;

// NOTE These functions are implemented using assembly because they use a custom
// calling convention which can't be implemented using a normal Rust function

// NOTE These functions are never mangled as they are not tested against compiler-rt

intrinsics! {
    #[unsafe(naked)]
    #[cfg(any(all(windows, target_env = "gnu"), target_os = "cygwin", target_os = "uefi"))]
    pub unsafe extern "custom" fn ___chkstk_ms() {
        core::arch::naked_asm!(
            "push   rcx",
            "push   rax",
            "cmp    rax, 0x1000",
            "lea    rcx, [rsp + 24]",
            "jb     3f",
            "2:",
            "sub    rcx, 0x1000",
            "test   [rcx], rcx",
            "sub    rax, 0x1000",
            "cmp    rax, 0x1000",
            "ja     2b",
            "3:",
            "sub    rcx, rax",
            "test   [rcx], rcx",
            "pop    rax",
            "pop    rcx",
            "ret",
        );
    }
}

// HACK(https://github.com/rust-lang/rust/issues/62785): x86_64-unknown-uefi needs special LLVM
// support unless we emit the _fltused
mod _fltused {
    #[unsafe(no_mangle)]
    #[used]
    #[cfg(target_os = "uefi")]
    static _fltused: i32 = 0;
}
