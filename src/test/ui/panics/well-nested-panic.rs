// run-pass
// needs-unwind

use std::panic::catch_unwind;

struct Bomb;
impl Drop for Bomb {
    fn drop(&mut self) {
        let _ = catch_unwind(|| panic!("bomb"));
    }
}

fn main() {
    let _ = catch_unwind(|| {
        let _bomb = Bomb;
        panic!("main");
    });
}
