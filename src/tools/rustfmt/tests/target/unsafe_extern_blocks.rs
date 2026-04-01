// See tracking issue for unsafe_extern_blocks
// https://github.com/rust-lang/rust/issues/123743

#![feature(unsafe_extern_blocks)]

safe static TEST1: i32;

unsafe extern "C" {
    safe static TEST2: i32;
    unsafe static TEST3: i32;
    static TEST4: i32;

    pub safe static TEST5: i32;
    pub unsafe static TEST6: i32;
    pub static TEST7: i32;

    safe fn test1(i: i32);
}
