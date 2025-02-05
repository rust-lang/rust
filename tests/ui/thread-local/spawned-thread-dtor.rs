//! Ensure that TLS destructors run on a spawned thread that
//! exits the process.
//@ run-pass
//@ needs-threads
//@ check-run-results

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Foo dtor");
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    FOO.with(|_| {});

    std::thread::spawn(|| {
        FOO.with(|_| {});
        std::process::exit(0);
    });

    loop {
        std::thread::park();
    }
}
