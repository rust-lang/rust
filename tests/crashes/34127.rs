//@ compile-flags: -g -Copt-level=0 -Z verify-llvm-ir
//@ known-bug: #34127
//@ only-64bit

pub fn main() {
let _a = [(); 1 << 63];
}
