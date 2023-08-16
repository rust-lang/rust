// run-fail
//@error-in-other-file:thread '<unnamed>' panicked at 'test'
//@ignore-target-emscripten Needs threads

use std::thread;

fn main() {
    let r: Result<(), _> = thread::spawn(move || {
                               panic!("test");
                           })
                               .join();
    assert!(r.is_ok());
}
