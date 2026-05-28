//! Regression test for https://github.com/rust-lang/rust/issues/14393

//@ run-pass

fn main() {
    match ("", 1_usize) {
        (_, 42_usize) => (),
        ("", _) => (),
        _ => ()
    }
}
