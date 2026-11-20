//@ run-pass
//@ needs-unwind
//@ needs-threads

// rust-lang/rust#64655: with panic=unwind, a panic from a subroutine
// should still run destructors as it unwinds the stack. However,
// bugs with how the nounwind LLVM attribute was applied led to this
// simple case being mishandled *if* you had fat LTO turned on.

// Unlike issue-64655-extern-rust-must-allow-unwind.rs, the issue
// embodied in this test cropped up regardless of optimization level.
// Therefore it seemed worthy of being enshrined as a dedicated unit
// test.

// LTO settings cannot be combined with -C prefer-dynamic
//@ no-prefer-dynamic

// The revisions just enumerate lto settings (the opt-level appeared irrelevant in practice)

//@ revisions: no thin fat
//@[no]compile-flags: -C lto=no
//@[thin]compile-flags: -C lto=thin
//@[fat]compile-flags: -C lto=fat
//@ ignore-backends: gcc

#![feature(panic_internals)]

// (For some reason, reproducing the LTO issue requires pulling in std
// explicitly this way.)
#![no_std]
extern crate std;

fn main() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::boxed::Box;

    static SHARED: AtomicUsize = AtomicUsize::new(0);

    assert_eq!(SHARED.fetch_add(0, Ordering::SeqCst), 0);

    let old_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(|_| { } )); // no-op on panic.

    let handle = std::thread::spawn(|| {
        struct Droppable;
        impl Drop for Droppable {
            fn drop(&mut self) {
                SHARED.fetch_add(1, Ordering::SeqCst);
            }
        }

        let _guard = Droppable;
        core::panicking::panic("???");
    });

    let wait = handle.join();

    // Reinstate handler to ease observation of assertion failures.
    std::panic::set_hook(old_hook);

    assert!(wait.is_err());

    assert_eq!(SHARED.fetch_add(0, Ordering::SeqCst), 1);
}
