// aux-build:test-macros.rs
// ignore-wasm32

#[macro_use]
extern crate test_macros;

fn main() {
    assert_eq!(unsafe { rust_get_test_int() }, 0isize);
    assert_eq!(unsafe { rust_dbg_extern_identity_u32(0xDEADBEEF) }, 0xDEADBEEF);
}

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    #[empty_attr]
    //~^ ERROR macro invocations in `extern {}` blocks are experimental
    fn some_definitely_unknown_symbol_which_should_be_removed();

    #[identity_attr]
    //~^ ERROR macro invocations in `extern {}` blocks are experimental
    fn rust_get_test_int() -> isize;

    identity!(fn rust_dbg_extern_identity_u32(arg: u32) -> u32;);
    //~^ ERROR macro invocations in `extern {}` blocks are experimental
}
