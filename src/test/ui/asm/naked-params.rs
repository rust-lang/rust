// Check that use of function parameters is validate in naked functions.
//
// ignore-wasm32 asm unsupported
#![feature(asm)]
#![feature(naked_functions)]
#![feature(or_patterns)]
#![crate_type = "lib"]

#[repr(C)]
pub struct P { x: u8, y: u16 }

#[naked]
pub unsafe extern "C" fn f(
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
    a + 1
    //~^ ERROR use of parameters not allowed inside naked functions
}

#[naked]
pub unsafe extern "C" fn inc_asm(a: u32) -> u32 {
    asm!("/* {0} */", in(reg) a, options(noreturn));
    //~^ ERROR use of parameters not allowed inside naked functions
}

#[naked]
pub unsafe extern "C" fn sum(x: u32, y: u32) -> u32 {
    // FIXME: Should be detected by asm-only check.
    (|| { x + y})()
}

pub fn outer(x: u32) -> extern "C" fn(usize) -> usize {
    #[naked]
    pub extern "C" fn inner(y: usize) -> usize {
        *&y
        //~^ ERROR use of parameters not allowed inside naked functions
    }
    inner
}
