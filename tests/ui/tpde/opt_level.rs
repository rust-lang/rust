//@ revisions: zero one two three
//@ needs-tpde
//@ build-pass
//@ compile-flags: -Zllvm-tpde
//
//@[zero] compile-flags: -Copt-level=0
//@[one] compile-flags: -Copt-level=1
//@[two] compile-flags: -Copt-level=2
//@[three] compile-flags: -Copt-level=3

//[two,three]~? WARN: using -Zllvm-tpde with optimization levels higher than 1 is not recommended
//[two,three]~? NOTE: compilation with TPDE at higher opt levels is more likely to fail, and to have worse runtime performance than LLVM even if it succeeds

fn main() {}
