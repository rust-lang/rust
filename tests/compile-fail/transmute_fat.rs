// This should fail even without validation
// compile-flags: -Zmir-emit-validate=0
#![feature(i128_type)]

fn main() {
    #[cfg(target_pointer_width="64")]
    let bad = unsafe {
        std::mem::transmute::<&[u8], u128>(&[1u8])
    };
    #[cfg(target_pointer_width="32")]
    let bad = unsafe {
        std::mem::transmute::<&[u8], u64>(&[1u8])
    };
    bad + 1; //~ ERROR a raw memory access tried to access part of a pointer value as raw bytes
}
