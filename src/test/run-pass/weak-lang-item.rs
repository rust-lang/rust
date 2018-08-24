// aux-build:weak-lang-items.rs

// ignore-emscripten no threads support
// pretty-expanded FIXME #23616

extern crate weak_lang_items as other;

use std::thread;

fn main() {
    let _ = thread::spawn(move|| {
        other::foo()
    });
}
