//! Regression test for <https://github.com/rust-lang/rust/issues/165290>:
//! thread-local storage can't be borrowed past the end of functions.

#![feature(thread_local)]

#[thread_local]
static TLS: [u8; 64 << 20] = [0; 64 << 20];

fn leak_tls() -> &'static [u8; 64 << 20] {
    &TLS //~ ERROR: thread-local variable borrowed past end of function
}

fn main() {
    let dangling = std::thread::spawn(leak_tls).join().unwrap();
    std::hint::black_box(dangling[0]);
}
