//@ run-fail
//@ check-run-results
//@ needs-threads

use std::thread;

fn main() {
    let r: Result<(), _> = thread::spawn(move || {
                               panic!("test");
                           })
                               .join();
    assert!(r.is_ok());
}
