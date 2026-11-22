//@ check-pass

#![crate_type = "lib"]
#![doc(test(no_crate_inject))]
#![doc(test(attr(deny(warnings))))]

pub fn foo() {}
