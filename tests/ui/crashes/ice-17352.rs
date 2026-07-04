//@ check-pass
#![warn(clippy::unnecessary_unwrap_unchecked)]

fn issue17352(x: impl Fn() -> Option<u32>) {
    _ = unsafe { x().unwrap_unchecked() };
}
