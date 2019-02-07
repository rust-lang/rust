#![crate_type = "cdylib"]

extern "C" {
    fn observe(ptr: *const u8, len: usize);

    fn get_u8() -> u8;
    fn get_i8() -> i8;
    fn get_u16() -> u16;
    fn get_i16() -> i16;
    fn get_u32() -> u32;
    fn get_i32() -> i32;
    fn get_u64() -> u64;
    fn get_i64() -> i64;
    fn get_usize() -> usize;
    fn get_isize() -> isize;
}

macro_rules! stringify {
    ( $($f:ident)* ) => {
        $(
            let s = $f().to_string();
            observe(s.as_ptr(), s.len());
        )*
    };
}

#[no_mangle]
pub unsafe extern "C" fn foo() {
    stringify!(get_u8);
    stringify!(get_i8);
    stringify!(get_u16);
    stringify!(get_i16);
    stringify!(get_u32);
    stringify!(get_i32);
    stringify!(get_u64);
    stringify!(get_i64);
    stringify!(get_usize);
    stringify!(get_isize);
}
