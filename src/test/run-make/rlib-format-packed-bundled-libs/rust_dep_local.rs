#[link(name = "native_dep_1", kind = "static")]
extern "C" {
    fn native_f1() -> i32;
}

extern crate rust_dep_up;

pub fn rust_dep_local() {
    unsafe {
        assert!(native_f1() == 1);
    }
    rust_dep_up::rust_dep_up();
}
