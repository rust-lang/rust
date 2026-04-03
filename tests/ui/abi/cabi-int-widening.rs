//@ run-pass

#[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
#[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
extern "C" {
    fn rust_int8_to_int32(_: i8) -> i32;
}

fn main() {
    let x = unsafe {
        rust_int8_to_int32(-1)
    };

    assert!(x == -1);
}
