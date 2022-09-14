extern crate rust_dep;

extern "C" {
    fn local_native_f() -> i32;
}

pub fn main() {
    unsafe {
        assert!(local_native_f() == 0);
    };
    rust_dep::rust_dep()
}
