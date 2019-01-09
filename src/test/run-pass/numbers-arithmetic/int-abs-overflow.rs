// run-pass
// compile-flags: -Z force-overflow-checks=on
// ignore-emscripten no threads support

use std::thread;

fn main() {
    assert!(thread::spawn(|| i8::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| i16::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| i32::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| i64::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| isize::min_value().abs()).join().is_err());
}
