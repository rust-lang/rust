// compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
// only-thumbv8m.main-none-eabi
#![feature(abi_c_cmse_nonsecure_call)]
#![no_std]

#[no_mangle]
pub fn test(a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
    let non_secure_function = unsafe {
        core::mem::transmute::<
            usize,
            extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u32, u32) -> u32>
        (
            0x10000004,
        )
    };
    non_secure_function(a, b, c, d, e)
}
