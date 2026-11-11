// Out-of-line module is found on the filesystem if passed through a proc macro (issue #58818).

//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

mod outer {
    identity! { mod inner; }
}

fn main() {}
