//! Regression test for https://github.com/rust-lang/rust/issues/16256

//@ run-pass

fn main() {
    let mut buf = Vec::new();
    |c: u8| buf.push(c); //~ WARN unused closure that must be used
}
