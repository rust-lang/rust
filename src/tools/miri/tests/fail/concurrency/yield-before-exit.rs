//! Regression test for https://github.com/rust-lang/miri/issues/2629:
//! If we fail to yield from the main thread, we miss the UB here.
//@compile-flags: -Zmiri-fixed-schedule
//@revisions: with_exit without_exit

fn main() {
    std::thread::spawn(|| unsafe { std::hint::unreachable_unchecked() });
    //~^ERROR: Undefined Behavior
    if cfg!(with_exit) {
        std::process::exit(0);
    }
}
