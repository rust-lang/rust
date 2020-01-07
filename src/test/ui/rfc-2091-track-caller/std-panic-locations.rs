// run-pass
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(option_expect_none, option_unwrap_none)]

//! Test that panic locations for `#[track_caller]` functions in std have the correct
//! location reported.

fn main() {
    // inspect the `PanicInfo` we receive to ensure the right file is the source
    std::panic::set_hook(Box::new(|info| {
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

    let yep: Option<()> = Some(());
    assert_panicked(|| yep.unwrap_none());
    assert_panicked(|| yep.expect_none(""));

    let oops: Result<(), ()> = Err(());
    assert_panicked(|| oops.unwrap());
    assert_panicked(|| oops.expect(""));

    let fine: Result<(), ()> = Ok(());
    assert_panicked(|| fine.unwrap_err());
    assert_panicked(|| fine.expect_err(""));
}
