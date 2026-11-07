// Tests that C++ double unwinding through Rust code will be properly guarded
// against instead of exhibiting undefined behaviour.

extern "C-unwind" {
    fn throw_cxx_exception();
    fn cxx_catch_callback(cb: extern "C-unwind" fn());
}

struct ThrowOnDrop;

impl Drop for ThrowOnDrop {
    fn drop(&mut self) {
        unsafe { throw_cxx_exception() };
    }
}

extern "C-unwind" fn test_double_unwind() {
    let _a = ThrowOnDrop;
    let _b = ThrowOnDrop;
}

fn main() {
    unsafe { cxx_catch_callback(test_double_unwind) };
}
