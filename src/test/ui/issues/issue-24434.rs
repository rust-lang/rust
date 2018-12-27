// compile-pass
#![allow(unused_attributes)]
// compile-flags:--cfg set1

#![cfg_attr(set1, feature(custom_attribute))]

#![foobar]
fn main() {}
