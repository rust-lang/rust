//@compile-flags: -Zmiri-strict-provenance

fn main() {
    let addr = &0 as *const i32 as usize;
    let _ptr = addr as *const i32; //~ ERROR integer-to-pointer casts and `ptr::from_exposed_addr` are not supported
}
