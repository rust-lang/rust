//@ run-pass

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn rust_int8_to_int32(_: i8) -> i32;
}

fn main() {
    let x = unsafe {
        rust_int8_to_int32(-1)
    };

    assert!(x == -1);
}
