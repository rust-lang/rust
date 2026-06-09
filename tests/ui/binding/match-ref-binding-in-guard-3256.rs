//@ run-pass

use std::sync::Mutex;

pub fn main() {
    let x = Some(Mutex::new(true));
    match x {
        Some(ref z) if *z.lock().unwrap() => {
            assert!(*z.lock().unwrap());
        },
        _ => panic!()
    }
}
