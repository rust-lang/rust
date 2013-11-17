#[crate_type = "rlib"];

pub static mut statik: int = 0;

struct A;
impl Drop for A {
    fn drop(&mut self) {
        unsafe { statik = 1; }
    }
}

pub fn callback(f: ||) {
    let _a = A;
    f();
}
