#![feature(intrinsics)]
extern "rust-intrinsic" {
    pub static breakpoint : unsafe extern "rust-intrinsic" fn();
    //~^ ERROR intrinsic must be a function [E0622]
}
fn main() { unsafe { breakpoint(); } }
