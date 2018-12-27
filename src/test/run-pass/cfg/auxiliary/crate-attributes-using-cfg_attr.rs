// no-prefer-dynamic
// compile-flags: --cfg foo

#![cfg_attr(foo, crate_type="lib")]

pub fn foo() {}
