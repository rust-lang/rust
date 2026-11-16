//@ revisions: off unspecified packed unpacked
//@ needs-tpde
//@ compile-flags: -Zllvm-tpde
//
//@[off] build-pass
//@[unspecified] build-pass
//@[packed] check-fail
//@[unpacked] check-fail
//
//@[off] compile-flags: -Csplit-debuginfo=off
//@[packed] compile-flags: -Csplit-debuginfo=packed
//@[unpacked] compile-flags: -Csplit-debuginfo=unpacked

//[packed,unpacked]~? ERROR: using -Zllvm-tpde with split debuginfo is not supported

fn main() {}
