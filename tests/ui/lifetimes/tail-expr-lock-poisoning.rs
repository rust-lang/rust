//@ revisions: edition2021 edition2024
//@ needs-subprocess
//@ [edition2024] edition: 2024
//@ run-pass
//@ needs-unwind

use std::sync::Mutex;

struct PanicOnDrop;
impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        panic!()
    }
}

fn f(m: &Mutex<i32>) -> i32 {
    let _x = PanicOnDrop;
    *m.lock().unwrap()
}

fn main() {
    let m = Mutex::new(0);
    let _ = std::panic::catch_unwind(|| f(&m));
    #[cfg(edition2024)]
    assert!(m.lock().is_ok());
    #[cfg(edition2021)]
    assert!(m.lock().is_err());
}
