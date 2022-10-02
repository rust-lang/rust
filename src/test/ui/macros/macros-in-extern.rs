// run-pass
// ignore-wasm32

#![feature(decl_macro)]

macro_rules! returns_isize(
    ($ident:ident) => (
        fn $ident() -> isize;
    )
);

macro takes_u32_returns_u32($ident:ident) {
    fn $ident(arg: u32) -> u32;
}

macro_rules! emits_nothing(
    () => ()
);

macro_rules! emits_multiple(
    () => {
        fn f1() -> u32;
        fn f2() -> u32;
    }
);

mod defs {
    #[no_mangle]
    extern "C" fn f1() -> u32 {
        1
    }
    #[no_mangle]
    extern "C" fn f2() -> u32 {
        2
    }
}

fn main() {
    assert_eq!(unsafe { rust_get_test_int() }, 1);
    assert_eq!(unsafe { rust_dbg_extern_identity_u32(0xDEADBEEF) }, 0xDEADBEEFu32);
    assert_eq!(unsafe { f1() }, 1);
    assert_eq!(unsafe { f2() }, 2);
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    returns_isize!(rust_get_test_int);
    takes_u32_returns_u32!(rust_dbg_extern_identity_u32);
    emits_nothing!();
    emits_multiple!();
}
