//@ revisions: implicit try only invalid
//@ needs-tpde
//
//@[implicit] build-pass
//@[try] build-pass
//@[only] build-pass
//@[invalid] check-fail
//
//@[implicit] compile-flags: -Zllvm-tpde
//@[try] compile-flags: -Zllvm-tpde=try
//@[only] compile-flags: -Zllvm-tpde=only
//@[invalid] compile-flags: -Zllvm-tpde=invalid

//[invalid]~? ERROR: incorrect value `invalid` for unstable option `llvm-tpde` - either no value or one of: `try` (default), `only` was expected

fn main() {}
