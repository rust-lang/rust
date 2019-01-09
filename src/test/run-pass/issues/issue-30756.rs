// run-pass
#![forbid(unsafe_code)]

thread_local!(static FOO: u8 = 1);

fn main() {
}
