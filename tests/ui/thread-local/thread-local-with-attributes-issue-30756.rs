//@ run-pass
#![forbid(unsafe_code)]

thread_local!(static FOO: u8 = 1);

fn main() {
}

// https://github.com/rust-lang/rust/issues/30756
