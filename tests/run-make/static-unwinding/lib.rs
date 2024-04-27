#![crate_type = "rlib"]

pub static mut statik: isize = 0;

struct A;
impl Drop for A {
    fn drop(&mut self) {
        unsafe { statik = 1; }
    }
}

pub fn callback<F>(f: F) where F: FnOnce() {
    let _a = A;
    f();
}
