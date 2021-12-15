// needs-asm-support
// ignore-nvptx64
// ignore-spirv
// ignore-wasm32

#![feature(llvm_asm)]
#![feature(naked_functions)]
#![feature(or_patterns)]
#![feature(asm_const, asm_sym)]
#![crate_type = "lib"]
#![allow(deprecated)] // llvm_asm!

use std::arch::asm;

#[repr(C)]
pub struct P {
    x: u8,
    y: u16,
}

#[naked]
pub unsafe extern "C" fn patterns(
    mut a: u32,
    //~^ ERROR patterns not allowed in naked function parameters
    &b: &i32,
    //~^ ERROR patterns not allowed in naked function parameters
    (None | Some(_)): Option<std::ptr::NonNull<u8>>,
    //~^ ERROR patterns not allowed in naked function parameters
    P { x, y }: P,
    //~^ ERROR patterns not allowed in naked function parameters
) {
    asm!("", options(noreturn))
}

#[naked]
pub unsafe extern "C" fn inc(a: u32) -> u32 {
    //~^ WARN naked functions must contain a single asm block
    //~| WARN this was previously accepted
    a + 1
    //~^ ERROR referencing function parameters is not allowed in naked functions
}

#[naked]
pub unsafe extern "C" fn inc_asm(a: u32) -> u32 {
    asm!("/* {0} */", in(reg) a, options(noreturn));
    //~^ ERROR referencing function parameters is not allowed in naked functions
    //~| WARN only `const` and `sym` operands are supported in naked functions
    //~| WARN this was previously accepted
}

#[naked]
pub unsafe extern "C" fn inc_closure(a: u32) -> u32 {
    //~^ WARN naked functions must contain a single asm block
    //~| WARN this was previously accepted
    (|| a + 1)()
}

#[naked]
pub unsafe extern "C" fn unsupported_operands() {
    //~^ WARN naked functions must contain a single asm block
    //~| WARN this was previously accepted
    let mut a = 0usize;
    let mut b = 0usize;
    let mut c = 0usize;
    let mut d = 0usize;
    let mut e = 0usize;
    const F: usize = 0usize;
    static G: usize = 0usize;
    asm!("/* {0} {1} {2} {3} {4} {5} {6} */",
         //~^ WARN asm in naked functions must use `noreturn` option
         //~| WARN this was previously accepted
         in(reg) a,
         //~^ WARN only `const` and `sym` operands are supported in naked functions
         //~| WARN this was previously accepted
         inlateout(reg) b,
         inout(reg) c,
         lateout(reg) d,
         out(reg) e,
         const F,
         sym G,
    );
}

#[naked]
pub extern "C" fn missing_assembly() {
    //~^ WARN naked functions must contain a single asm block
    //~| WARN this was previously accepted
}

#[naked]
pub extern "C" fn too_many_asm_blocks() {
    //~^ WARN naked functions must contain a single asm block
    //~| WARN this was previously accepted
    asm!("");
    //~^ WARN asm in naked functions must use `noreturn` option
    //~| WARN this was previously accepted
    asm!("");
    //~^ WARN asm in naked functions must use `noreturn` option
    //~| WARN this was previously accepted
    asm!("");
    //~^ WARN asm in naked functions must use `noreturn` option
    //~| WARN this was previously accepted
    asm!("", options(noreturn));
}

pub fn outer(x: u32) -> extern "C" fn(usize) -> usize {
    #[naked]
    pub extern "C" fn inner(y: usize) -> usize {
        //~^ WARN naked functions must contain a single asm block
        //~| WARN this was previously accepted
        *&y
        //~^ ERROR referencing function parameters is not allowed in naked functions
    }
    inner
}

#[naked]
unsafe extern "C" fn llvm() -> ! {
    //~^ WARN naked functions must contain a single asm block
    //~| WARN this was previously accepted
    llvm_asm!("");
    //~^ WARN LLVM-style inline assembly is unsupported in naked functions
    //~| WARN this was previously accepted
    core::hint::unreachable_unchecked();
}

#[naked]
unsafe extern "C" fn invalid_options() {
    asm!("", options(nomem, preserves_flags, noreturn));
    //~^ WARN asm options unsupported in naked functions: `nomem`, `preserves_flags`
    //~| WARN this was previously accepted
}

#[naked]
unsafe extern "C" fn invalid_options_continued() {
    asm!("", options(readonly, nostack), options(pure));
    //~^ ERROR asm with the `pure` option must have at least one output
    //~| WARN asm options unsupported in naked functions: `nostack`, `pure`, `readonly`
    //~| WARN this was previously accepted
    //~| WARN asm in naked functions must use `noreturn` option
    //~| WARN this was previously accepted
}

#[naked]
pub unsafe fn default_abi() {
    //~^ WARN Rust ABI is unsupported in naked functions
    asm!("", options(noreturn));
}

#[naked]
pub unsafe fn rust_abi() {
    //~^ WARN Rust ABI is unsupported in naked functions
    asm!("", options(noreturn));
}

#[naked]
pub extern "C" fn valid_a<T>() -> T {
    unsafe {
        asm!("", options(noreturn));
    }
}

#[naked]
pub extern "C" fn valid_b() {
    unsafe {
        {
            {
                asm!("", options(noreturn));
            };
        };
    }
}

#[naked]
pub unsafe extern "C" fn valid_c() {
    asm!("", options(noreturn));
}

#[cfg(target_arch = "x86_64")]
#[naked]
pub unsafe extern "C" fn valid_att_syntax() {
    asm!("", options(noreturn, att_syntax));
}

#[naked]
pub unsafe extern "C" fn inline_none() {
    asm!("", options(noreturn));
}

#[naked]
#[inline]
//~^ WARN naked functions cannot be inlined
//~| WARN this was previously accepted
pub unsafe extern "C" fn inline_hint() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(always)]
//~^ WARN naked functions cannot be inlined
//~| WARN this was previously accepted
pub unsafe extern "C" fn inline_always() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(never)]
//~^ WARN naked functions cannot be inlined
//~| WARN this was previously accepted
pub unsafe extern "C" fn inline_never() {
    asm!("", options(noreturn));
}

#[naked]
#[inline]
//~^ WARN naked functions cannot be inlined
//~| WARN this was previously accepted
#[inline(always)]
//~^ WARN naked functions cannot be inlined
//~| WARN this was previously accepted
#[inline(never)]
//~^ WARN naked functions cannot be inlined
//~| WARN this was previously accepted
pub unsafe extern "C" fn inline_all() {
    asm!("", options(noreturn));
}
