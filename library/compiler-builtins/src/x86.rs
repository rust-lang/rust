#![allow(unused_imports)]

use core::intrinsics;

// NOTE These functions are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function

// NOTE These functions are never mangled as they are not tested against compiler-rt
// and mangling ___chkstk would break the `jmp ___chkstk` instruction in __alloca

#[cfg(all(
    windows,
    target_env = "gnu",
    not(feature = "no-asm"),
    not(feature = "mangled-names")
))]
#[naked]
#[no_mangle]
pub unsafe fn ___chkstk_ms() {
    llvm_asm!("
        push   %ecx
        push   %eax
        cmp    $$0x1000,%eax
        lea    12(%esp),%ecx
        jb     1f
    2:
        sub    $$0x1000,%ecx
        test   %ecx,(%ecx)
        sub    $$0x1000,%eax
        cmp    $$0x1000,%eax
        ja     2b
    1:
        sub    %eax,%ecx
        test   %ecx,(%ecx)
        pop    %eax
        pop    %ecx
        ret" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

// FIXME: __alloca should be an alias to __chkstk
#[cfg(all(
    windows,
    target_env = "gnu",
    not(feature = "no-asm"),
    not(feature = "mangled-names")
))]
#[naked]
#[no_mangle]
pub unsafe fn __alloca() {
    llvm_asm!("jmp ___chkstk   // Jump to ___chkstk since fallthrough may be unreliable"
         ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(all(
    windows,
    target_env = "gnu",
    not(feature = "no-asm"),
    not(feature = "mangled-names")
))]
#[naked]
#[no_mangle]
pub unsafe fn ___chkstk() {
    llvm_asm!("
        push   %ecx
        cmp    $$0x1000,%eax
        lea    8(%esp),%ecx     // esp before calling this routine -> ecx
        jb     1f
    2:
        sub    $$0x1000,%ecx
        test   %ecx,(%ecx)
        sub    $$0x1000,%eax
        cmp    $$0x1000,%eax
        ja     2b
    1:
        sub    %eax,%ecx
        test   %ecx,(%ecx)

        lea    4(%esp),%eax     // load pointer to the return address into eax
        mov    %ecx,%esp        // install the new top of stack pointer into esp
        mov    -4(%eax),%ecx    // restore ecx
        push   (%eax)           // push return address onto the stack
        sub    %esp,%eax        // restore the original value in eax
        ret" ::: "memory" : "volatile");
    intrinsics::unreachable();
}
