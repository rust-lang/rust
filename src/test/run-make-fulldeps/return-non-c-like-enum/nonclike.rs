#![crate_type = "lib"]
#![crate_name = "nonclike"]

#[repr(C,u8)]
pub enum T {
    A(u64),
    B,
}

#[no_mangle]
pub extern "C" fn t_new(a: u64) -> T {
    T::A(a)
}
