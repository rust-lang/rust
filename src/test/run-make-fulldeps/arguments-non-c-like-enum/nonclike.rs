#![crate_type = "lib"]
#![crate_name = "nonclike"]

#[repr(C,u8)]
pub enum T {
    A(u64),
    B,
}

#[no_mangle]
pub extern "C" fn t_add(a: T, b: T) -> u64 {
    match (a,b) {
        (T::A(a), T::A(b)) => a + b,
        (T::A(a), T::B) => a,
        (T::B, T::A(b)) => b,
        _ => 0,
    }
}
