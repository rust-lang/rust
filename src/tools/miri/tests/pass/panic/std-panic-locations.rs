//! Test that panic locations for `#[track_caller]` functions in std have the correct
//! location reported.

use std::sync::atomic::{AtomicUsize, Ordering};

static HOOK_COUNT: AtomicUsize = AtomicUsize::new(0);

fn main() {
    // inspect the `PanicInfo` we receive to ensure the right file is the source
    std::panic::set_hook(Box::new(|info| {
        HOOK_COUNT.fetch_add(1, Ordering::Relaxed);
        let actual = info.location().unwrap();
        if actual.file() != file!() {
            eprintln!("expected a location in the test file, found {:?}", actual);
            panic!();
        }
    }));

    fn assert_panicked(f: impl FnOnce() + std::panic::UnwindSafe) {
        std::panic::catch_unwind(f).unwrap_err();
    }

    let nope: Option<()> = None;
    assert_panicked(|| nope.unwrap());
    assert_panicked(|| nope.expect(""));

    let oops: Result<(), ()> = Err(());
    assert_panicked(|| oops.unwrap());
    assert_panicked(|| oops.expect(""));

    let fine: Result<(), ()> = Ok(());
    assert_panicked(|| fine.unwrap_err());
    assert_panicked(|| fine.expect_err(""));

    // Cleanup: reset to default hook.
    drop(std::panic::take_hook());

    assert_eq!(HOOK_COUNT.load(Ordering::Relaxed), 6);
}
