// build-pass
//
// compile-flags: -g --emit=llvm-ir -Zunstable-options -Csplit-debuginfo=unpacked
//
// Make sure that we don't explode with an error if we don't actually end up emitting any `dwo`s,
// as would be the case if we don't actually codegen anything.
#![crate_type="rlib"]
