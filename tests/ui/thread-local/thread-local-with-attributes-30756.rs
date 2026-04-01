// https://github.com/rust-lang/rust/issues/30756
//@ run-pass
#![forbid(unsafe_code)]

thread_local!(static FOO: u8 = 1);

fn main() {
}
