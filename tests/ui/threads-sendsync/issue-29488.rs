//@ run-pass
//@ needs-threads

use std::thread;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("test2");
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    // Off the main thread due to #28129, be sure to initialize FOO first before
    // calling `println!`
    thread::spawn(|| {
        FOO.with(|_| {});
        println!("test1");
    })
    .join()
    .unwrap();
}
