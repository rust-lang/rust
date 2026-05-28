//@ run-pass
//@ aux-build:foreign_lib.rs

// Check that we can still call duplicated extern (imported) functions
// which were declared in another crate. See issues #32740 and #32783.

extern crate foreign_lib;

pub fn main() {
    unsafe {
        let x = foreign_lib::rustrt::rust_get_test_int();
        assert_eq!(x, foreign_lib::rustrt2::rust_get_test_int());
        assert_eq!(x as *const u8, foreign_lib::rustrt3::rust_get_test_int());
    }
}
