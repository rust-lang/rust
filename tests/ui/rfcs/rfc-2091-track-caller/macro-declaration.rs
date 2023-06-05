// check-pass

// See https://github.com/rust-lang/rust/issues/95151
#[track_caller]
macro_rules! _foo {
    () => {};
}

fn main() {
}
