//@ check-fail
//@ revisions: stable unstable
//@ [unstable] compile-flags: -Z stack-protector=basic
//@ [stable] compile-flags: -C stack-protector=basic

pub fn main(){}

//[unstable]~? ERROR incorrect value `basic` for unstable option `stack-protector`
//[stable]~? ERROR incorrect value `basic` for codegen option `stack-protector`
