//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

#![feature(naked_functions)]
#![feature(asm_const, asm_unwind)]
#![crate_type = "lib"]

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
    //~^ ERROR naked functions must contain a single asm block
    a + 1
    //~^ ERROR referencing function parameters is not allowed in naked functions
}

#[naked]
#[allow(asm_sub_register)]
pub unsafe extern "C" fn inc_asm(a: u32) -> u32 {
    asm!("/* {0} */", in(reg) a, options(noreturn));
    //~^ ERROR referencing function parameters is not allowed in naked functions
    //~| ERROR only `const` and `sym` operands are supported in naked functions
}

#[naked]
pub unsafe extern "C" fn inc_closure(a: u32) -> u32 {
    //~^ ERROR naked functions must contain a single asm block
    (|| a + 1)()
}

#[naked]
pub unsafe extern "C" fn unsupported_operands() {
    //~^ ERROR naked functions must contain a single asm block
    let mut a = 0usize;
    let mut b = 0usize;
    let mut c = 0usize;
    let mut d = 0usize;
    let mut e = 0usize;
    const F: usize = 0usize;
    static G: usize = 0usize;
    asm!("/* {0} {1} {2} {3} {4} {5} {6} */",
         //~^ ERROR asm in naked functions must use `noreturn` option
         in(reg) a,
         //~^ ERROR only `const` and `sym` operands are supported in naked functions
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
    //~^ ERROR naked functions must contain a single asm block
}

#[naked]
pub extern "C" fn too_many_asm_blocks() {
    //~^ ERROR naked functions must contain a single asm block
    unsafe {
        asm!("");
        //~^ ERROR asm in naked functions must use `noreturn` option
        asm!("");
        //~^ ERROR asm in naked functions must use `noreturn` option
        asm!("");
        //~^ ERROR asm in naked functions must use `noreturn` option
        asm!("", options(noreturn));
    }
}

pub fn outer(x: u32) -> extern "C" fn(usize) -> usize {
    #[naked]
    pub extern "C" fn inner(y: usize) -> usize {
        //~^ ERROR naked functions must contain a single asm block
        *&y
        //~^ ERROR referencing function parameters is not allowed in naked functions
    }
    inner
}

#[naked]
unsafe extern "C" fn invalid_options() {
    asm!("", options(nomem, preserves_flags, noreturn));
    //~^ ERROR asm options unsupported in naked functions: `nomem`, `preserves_flags`
}

#[naked]
unsafe extern "C" fn invalid_options_continued() {
    asm!("", options(readonly, nostack), options(pure));
    //~^ ERROR asm with the `pure` option must have at least one output
    //~| ERROR asm options unsupported in naked functions: `pure`, `readonly`, `nostack`
    //~| ERROR asm in naked functions must use `noreturn` option
}

#[naked]
unsafe extern "C" fn invalid_may_unwind() {
    asm!("", options(noreturn, may_unwind));
    //~^ ERROR asm options unsupported in naked functions: `may_unwind`
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
//~^ ERROR naked functions cannot be inlined
pub unsafe extern "C" fn inline_hint() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(always)]
//~^ ERROR naked functions cannot be inlined
pub unsafe extern "C" fn inline_always() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(never)]
//~^ ERROR naked functions cannot be inlined
pub unsafe extern "C" fn inline_never() {
    asm!("", options(noreturn));
}

#[naked]
#[inline]
//~^ ERROR naked functions cannot be inlined
#[inline(always)]
//~^ ERROR naked functions cannot be inlined
#[inline(never)]
//~^ ERROR naked functions cannot be inlined
pub unsafe extern "C" fn inline_all() {
    asm!("", options(noreturn));
}

#[naked]
pub unsafe extern "C" fn allow_compile_error(a: u32) -> u32 {
    compile_error!("this is a user specified error")
    //~^ ERROR this is a user specified error
}

#[naked]
pub unsafe extern "C" fn allow_compile_error_and_asm(a: u32) -> u32 {
    compile_error!("this is a user specified error");
    //~^ ERROR this is a user specified error
    asm!("", options(noreturn))
}

#[naked]
pub unsafe extern "C" fn invalid_asm_syntax(a: u32) -> u32 {
    asm!(invalid_syntax)
    //~^ ERROR asm template must be a string literal
}
