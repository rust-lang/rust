#![allow(unused_imports)]

use core::intrinsics;

// NOTE These functions are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function

// NOTE These functions are never mangled as they are not tested against compiler-rt
// and mangling ___chkstk would break the `jmp ___chkstk` instruction in __alloca

intrinsics! {
    #[naked]
    #[cfg(all(
        windows,
        target_env = "gnu",
        not(feature = "no-asm")
    ))]
    pub unsafe extern "C" fn ___chkstk_ms() {
        core::arch::asm!(
            "push   %rcx",
            "push   %rax",
            "cmp    $0x1000,%rax",
            "lea    24(%rsp),%rcx",
            "jb     1f",
            "2:",
            "sub    $0x1000,%rcx",
            "test   %rcx,(%rcx)",
            "sub    $0x1000,%rax",
            "cmp    $0x1000,%rax",
            "ja     2b",
            "1:",
            "sub    %rax,%rcx",
            "test   %rcx,(%rcx)",
            "pop    %rax",
            "pop    %rcx",
            "ret",
            options(noreturn, att_syntax)
        );
    }

    #[naked]
    #[cfg(all(
        windows,
        target_env = "gnu",
        not(feature = "no-asm")
    ))]
    pub unsafe extern "C" fn __alloca() {
        core::arch::asm!(
            "mov    %rcx,%rax", // x64 _alloca is a normal function with parameter in rcx
            "jmp    ___chkstk", // Jump to ___chkstk since fallthrough may be unreliable"
            options(noreturn, att_syntax)
        );
    }

    #[naked]
    #[cfg(all(
        windows,
        target_env = "gnu",
        not(feature = "no-asm")
    ))]
    pub unsafe extern "C" fn ___chkstk() {
        core::arch::asm!(
            "push   %rcx",
            "cmp    $0x1000,%rax",
            "lea    16(%rsp),%rcx", // rsp before calling this routine -> rcx
            "jb     1f",
            "2:",
            "sub    $0x1000,%rcx",
            "test   %rcx,(%rcx)",
            "sub    $0x1000,%rax",
            "cmp    $0x1000,%rax",
            "ja     2b",
            "1:",
            "sub    %rax,%rcx",
            "test   %rcx,(%rcx)",
            "lea    8(%rsp),%rax",  // load pointer to the return address into rax
            "mov    %rcx,%rsp",     // install the new top of stack pointer into rsp
            "mov    -8(%rax),%rcx", // restore rcx
            "push   (%rax)",        // push return address onto the stack
            "sub    %rsp,%rax",     // restore the original value in rax
            "ret",
            options(noreturn, att_syntax)
        );
    }
}

// HACK(https://github.com/rust-lang/rust/issues/62785): x86_64-unknown-uefi needs special LLVM
// support unless we emit the _fltused
mod _fltused {
    #[no_mangle]
    #[used]
    #[cfg(target_os = "uefi")]
    static _fltused: i32 = 0;
}
