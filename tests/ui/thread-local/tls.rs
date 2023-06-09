// run-pass
// ignore-emscripten no threads support
// compile-flags: -O
// ignore-nto Doesn't work without emulated TLS enabled (in LLVM)

#![feature(thread_local)]

#[thread_local]
static S: u32 = 222;

fn main() {
    let local = &S as *const u32 as usize;
    let foreign = std::thread::spawn(|| &S as *const u32 as usize).join().unwrap();
    assert_ne!(local, foreign);
}
