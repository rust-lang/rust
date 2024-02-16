// Test panic error messages for unnamed threads

// run-fail
// regex-error-pattern:thread '<unnamed>' \(id \d+\) panicked
// error-pattern:test
// ignore-emscripten Needs threads

use std::thread;

fn main() {
    let _: () = thread::spawn(move || {
        panic!("test");
    })
    .join()
    .unwrap();
}
