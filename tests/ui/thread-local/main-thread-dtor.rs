//! Ensure that TLS destructors run on the main thread.
//@ run-pass
//@ check-run-results

struct Bar;

impl Drop for Bar {
    fn drop(&mut self) {
        println!("Bar dtor");
    }
}

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Foo dtor");
        // We initialize another thread-local inside the dtor, which is an interesting corner case.
        thread_local!(static BAR: Bar = Bar);
        BAR.with(|_| {});
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    FOO.with(|_| {});
}
