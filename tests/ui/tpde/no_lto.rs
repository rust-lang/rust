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

//[thin]~? WARN: ThinLTO is not supported by the codegen backend, using fat LTO instead
//[thin,fat]~? ERROR: required feature `fat LTO` not supported by codegen backend

fn main() {}
