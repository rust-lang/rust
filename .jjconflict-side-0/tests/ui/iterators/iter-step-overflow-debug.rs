//@ run-pass
//@ needs-unwind
//@ compile-flags: -C debug_assertions=yes

use std::panic;

fn main() {
    let r = panic::catch_unwind(|| {
        let mut it = u8::MAX..;
        it.next().unwrap(); // 255
        it.next().unwrap();
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        let mut it = i8::MAX..;
        it.next().unwrap(); // 127
        it.next().unwrap();
    });
    assert!(r.is_err());
}
