extern "C" {
    fn local_native_f() -> i32;
}

pub fn main() {
    unsafe {
        assert!(local_native_f() == 0);
    };
}
