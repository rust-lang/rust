//@ check-pass

// Make sure we don't create an opaque def id for ITIB.

#![crate_type = "lib"]
#![feature(impl_trait_in_bindings)]

fn foo() {
    let _: impl Sized = 0;
}
