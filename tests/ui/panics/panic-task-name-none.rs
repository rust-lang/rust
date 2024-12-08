//@ run-fail
//@ error-pattern:thread '<unnamed>' panicked
//@ error-pattern:test
//@ needs-threads

use std::thread;

fn main() {
    let r: Result<(), _> = thread::spawn(move || {
                               panic!("test");
                           })
                               .join();
    assert!(r.is_ok());
}
