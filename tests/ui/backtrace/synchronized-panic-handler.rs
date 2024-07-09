//@ run-pass
//@ check-run-results
//@ edition:2021
use std::thread;
const PANIC_MESSAGE: &str = "oops oh no woe is me";

fn entry() {
    panic!("{PANIC_MESSAGE}")
}

fn main() {
    let (a, b) = (thread::spawn(entry), thread::spawn(entry));
    assert_eq!(&**a.join().unwrap_err().downcast::<String>().unwrap(), PANIC_MESSAGE);
    assert_eq!(&**b.join().unwrap_err().downcast::<String>().unwrap(), PANIC_MESSAGE);
}
