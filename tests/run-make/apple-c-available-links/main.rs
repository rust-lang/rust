unsafe extern "C" {
    safe fn foo() -> core::ffi::c_int;
}

fn main() {
    assert_eq!(foo(), 0);
}
