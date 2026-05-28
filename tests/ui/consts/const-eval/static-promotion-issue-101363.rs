//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/101363>

const OPTIONAL_SLICE_V1: Option<&'static [u8]> = Some(&{
    let array = [1, 2, 3];
    array
});

fn main() {
    let _ = OPTIONAL_SLICE_V1;
}
