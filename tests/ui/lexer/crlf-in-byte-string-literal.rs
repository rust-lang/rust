//! Regression test for <https://github.com/rust-lang/rust/issues/16278>.

//@ run-pass
// ignore-tidy-cr

// this file has some special \r\n endings (use xxd to see them)

fn main() {assert_eq!(b"", b"\
                                   ");
assert_eq!(b"\n", b"
");
}
