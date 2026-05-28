//@ check-pass
//@ edition: 2021
//
// Anti-regression test for https://github.com/rust-lang/rust/issues/140602
// where the generated warning couldn't be allowed due too being attached to
// the wrong AST node.

#![deny(unsafe_attr_outside_unsafe)]

#[allow(unsafe_attr_outside_unsafe)]
mod generated {
    #[no_mangle]
    fn _generated_foo() {}
}

fn main() {}
