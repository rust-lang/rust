//! Regression test for <https://github.com/rust-lang/rust/issues/23311>.
//! Test that we do not ICE when pattern matching an array against a slice.
//@ run-pass

fn main() {
    match "foo".as_bytes() {
        b"food" => (),
        &[b'f', ..] => (),
        _ => ()
    }
}
