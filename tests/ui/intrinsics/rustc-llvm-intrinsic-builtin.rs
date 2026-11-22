//@ run-pass

#![feature(intrinsics)]

#[rustc_intrinsic(llvm = "llvm.donothing")]
fn breakpoint() {}

fn main() {
    breakpoint();
}
