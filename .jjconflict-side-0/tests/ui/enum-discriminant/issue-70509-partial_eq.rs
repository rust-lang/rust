//@ run-pass

#[derive(PartialEq, Debug)]
#[repr(i128)]
enum Test {
    A(Box<u64>) = 0,
    B(usize) = u64::MAX as i128 + 1,
}

fn main() {
    assert_ne!(Test::A(Box::new(2)), Test::B(0));
    // This previously caused a segfault.
    //
    // See https://github.com/rust-lang/rust/issues/70509#issuecomment-620654186
    // for a detailed explanation.
}
