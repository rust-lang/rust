//! Regression test for https://github.com/rust-lang/rust/issues/42148
//@ run-pass
struct Zst;

fn main() {
    unsafe { ::std::ptr::write_volatile(1 as *mut Zst, Zst) }
}
