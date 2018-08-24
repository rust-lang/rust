// error-pattern:thread '<unnamed>' panicked at 'test'
// ignore-emscripten Needs threads

use std::thread;

fn main() {
    let r: Result<(), _> = thread::spawn(move || {
                               panic!("test");
                           })
                               .join();
    assert!(r.is_ok());
}
