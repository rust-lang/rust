// build-fail
// compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
// needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, no_core, lang_items, intrinsics)]
#![no_core]
#[lang="sized"]
pub trait Sized { }
#[lang="copy"]
pub trait Copy { }

extern "rust-intrinsic" {
    pub fn transmute<T, U>(e: T) -> U;
}

#[no_mangle]
pub fn test(a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
    let non_secure_function = unsafe {
        transmute::<
            usize,
            extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u32, u32) -> u32>
        (
            0x10000004,
        )
    };
    non_secure_function(a, b, c, d, e)
}
