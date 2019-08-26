// compile-flags: -O

#![crate_type = "lib"]

// The `nounwind` attribute does not get added by rustc; it is present here because LLVM
// analyses determine that this function does not unwind.

// CHECK: Function Attrs: norecurse nounwind
pub extern fn foo() {}
