//@ compile-flags: -g -Copt-level=0 -Z verify-llvm-ir
//@ known-bug: #34127
//@ only-x86_64

pub fn main() {
let _a = [(); 1 << 63];
}
