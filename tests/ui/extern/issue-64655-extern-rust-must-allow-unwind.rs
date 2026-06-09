//@ run-pass
//@ needs-unwind
//@ needs-threads

// rust-lang/rust#64655: with panic=unwind, a panic from a subroutine
// should still run destructors as it unwinds the stack. However,
// bugs with how the nounwind LLVM attribute was applied led to this
// simple case being mishandled *if* you had optimization *and* fat
// LTO turned on.

// This test is the closest thing to a "regression test" we can do
// without actually spawning subprocesses and comparing stderr
// results.
//
// This test takes the code from the above issue and adapts it to
// better fit our test infrastructure:
//
// * Instead of relying on `println!` to observe whether the destructor
//   is run, we instead run the code in a spawned thread and
//   communicate the destructor's operation via a synchronous atomic
//   in static memory.
//
// * To keep the output from confusing a casual user, we override the
//   panic hook to be a no-op (rather than printing a message to
//   stderr).
//
// (pnkfelix has confirmed by hand that these additions do not mask
// the underlying bug.)

// LTO settings cannot be combined with -C prefer-dynamic
//@ no-prefer-dynamic

// The revisions combine each lto setting with each optimization
// setting; pnkfelix observed three differing behaviors at opt-levels
// 0/1/2+3 for this test, so it seems prudent to be thorough.

//@ revisions: no0 no1 no2 no3 thin0 thin1 thin2 thin3 fat0 fat1 fat2  fat3

//@[no0]compile-flags: -C opt-level=0 -C lto=no
//@[no1]compile-flags: -C opt-level=1 -C lto=no
//@[no2]compile-flags: -C opt-level=2 -C lto=no
//@[no3]compile-flags: -C opt-level=3 -C lto=no
//@[thin0]compile-flags: -C opt-level=0 -C lto=thin
//@[thin1]compile-flags: -C opt-level=1 -C lto=thin
//@[thin2]compile-flags: -C opt-level=2 -C lto=thin
//@[thin3]compile-flags: -C opt-level=3 -C lto=thin
//@[fat0]compile-flags: -C opt-level=0 -C lto=fat
//@[fat1]compile-flags: -C opt-level=1 -C lto=fat
//@[fat2]compile-flags: -C opt-level=2 -C lto=fat
//@[fat3]compile-flags: -C opt-level=3 -C lto=fat
//@ ignore-backends: gcc

fn main() {
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        None::<()>.expect("???");
    });

    let wait = handle.join();

    // reinstate handler to ease observation of assertion failures.
    std::panic::set_hook(old_hook);

    assert!(wait.is_err());

    assert_eq!(SHARED.fetch_add(0, Ordering::SeqCst), 1);
}
