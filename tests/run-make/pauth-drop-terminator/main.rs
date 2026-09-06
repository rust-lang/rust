extern "C" {
    fn c_cleanup(x: i32);
}

struct Bomb(i32);

impl Drop for Bomb {
    fn drop(&mut self) {
        unsafe {
            c_cleanup(self.0);
        }
    }
}

pub fn may_unwind(x: i32) {
    let b = Bomb(x);

    match b.0 {
        0 => return,
        _ => {}
    }
}
