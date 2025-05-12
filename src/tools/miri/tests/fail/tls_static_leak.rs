//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"

#![feature(thread_local)]

use std::cell::Cell;

/// Ensure that leaks through `thread_local` statics *not in the main thread*
/// are detected.
pub fn main() {
    #[thread_local]
    static TLS: Cell<Option<&'static i32>> = Cell::new(None);

    std::thread::spawn(|| {
        TLS.set(Some(Box::leak(Box::new(123)))); //~ERROR: memory leaked
    })
    .join()
    .unwrap();

    // Imagine the program running for a long time while the thread is gone
    // and this memory still sits around, unused -- leaked.
}
