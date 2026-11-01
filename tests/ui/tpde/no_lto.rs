//@ revisions: no unspecified thin fat
//@ needs-tpde
//@ compile-flags: -Zllvm-tpde
//
//@[no] build-pass
//@[unspecified] build-pass
//@[thin] check-fail
//@[fat] check-fail
//
//@[no] compile-flags: -Clto=no
//@[thin] compile-flags: -Clto=thin
//@[fat] compile-flags: -Clto=fat

//[thin,fat]~? ERROR: using -Zllvm-tpde with LTO enabled is not supported

fn main() {}
