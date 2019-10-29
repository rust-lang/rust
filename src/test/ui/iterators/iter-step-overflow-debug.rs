// run-pass
// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -C debug_assertions=yes

use std::panic;

fn main() {
    let r = panic::catch_unwind(|| {
        let mut it = u8::max_value()..;
        it.next().unwrap(); // 255
        it.next().unwrap();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        let mut it = i8::max_value()..;
        it.next().unwrap(); // 127
        it.next().unwrap();
    });
    assert!(r.is_err());
}
