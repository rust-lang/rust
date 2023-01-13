//@compile-flags: -Zmiri-strict-provenance
#![feature(strict_provenance)]

fn main() {
    let addr = &0 as *const i32 as usize;
    let _ptr = std::ptr::from_exposed_addr::<i32>(addr); //~ ERROR: integer-to-pointer casts and `ptr::from_exposed_addr` are not supported
}
