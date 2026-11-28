use std::cell::Cell;
use std::{panic, thread};

unsafe extern "C-unwind" {
    safe fn caught_panic() -> bool;
}

thread_local! {
    static DROPPED: Cell<bool> = const { Cell::new(false) };
}

struct MarkDropped;

impl Drop for MarkDropped {
    fn drop(&mut self) {
        // Test that the panic count is reset before the panic payload is dropped.
        assert!(!thread::panicking());
        DROPPED.set(true);
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn rust_panic() {
    panic::resume_unwind(Box::new(MarkDropped));
}

fn main() {
    assert!(caught_panic());
    // Test that the payload is dropped...
    assert!(DROPPED.get());
    // ... and the panic count reset.
    assert!(!thread::panicking());
}
