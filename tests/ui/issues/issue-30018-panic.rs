// run-pass
// Regression test for Issue #30018. This is very similar to the
// original reported test, except that the panic is wrapped in a
// spawned thread to isolate the expected error result from the
// SIGTRAP injected by the drop-flag consistency checking.

// needs-unwind
// ignore-emscripten no threads support

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn foo() -> Foo {
    panic!();
}

fn main() {
    use std::thread;
    let handle = thread::spawn(|| {
        let _ = &[foo()];
    });
    let _ = handle.join();
}
