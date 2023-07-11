// run-pass
// aux-build:weak-lang-items.rs

// ignore-emscripten no threads support
// pretty-expanded FIXME #23616

extern crate weak_lang_items as other;

fn main() {
    let _ = std::thread::spawn(move || {
        // The goal of the test is just to make sure other::foo() is called. Since the function
        // panics, it's executed in its own thread. That way, the panic is isolated within the
        // thread and wont't affect the overall exit code.
        //
        // That causes a spurious failures in panic=abort targets though: if the program exits
        // before the thread is fully initialized the test will pass, but if the thread gets
        // executed first the whole program will abort. Adding a 60 seconds sleep will (hopefully!)
        // ensure the program always exits before the thread is executed.
        std::thread::sleep(std::time::Duration::from_secs(60));

        other::foo()
    });
}
