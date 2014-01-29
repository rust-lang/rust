extern mod lib;

use std::task;

static mut statik: int = 0;

struct A;
impl Drop for A {
    fn drop(&mut self) {
        unsafe { statik = 1; }
    }
}

fn main() {
    task::try(proc() {
        let _a = A;
        lib::callback(|| fail!());
        1
    });

    unsafe {
        assert!(lib::statik == 1);
        assert!(statik == 1);
    }
}
