#![crate_type = "cdylib"]

extern "C" {
    fn observe(ptr: *const u8, len: usize);
}

macro_rules! s {
    ( $( $f:ident -> $t:ty );* $(;)* ) => {
        $(
            extern "C" {
                fn $f() -> $t;
            }
            let s = $f().to_string();
            observe(s.as_ptr(), s.len());
        )*
    };
}

#[no_mangle]
pub unsafe extern "C" fn foo() {
    s! {
        get_u8 -> u8;
        get_i8 -> i8;
        get_u16 -> u16;
        get_i16 -> i16;
        get_u32 -> u32;
        get_i32 -> i32;
        get_u64 -> u64;
        get_i64 -> i64;
        get_usize -> usize;
        get_isize -> isize;
    }
}
