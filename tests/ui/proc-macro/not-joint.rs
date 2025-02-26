//@ run-pass
//@ proc-macro: not-joint.rs

extern crate not_joint as bar;
use bar::{tokens, nothing};

tokens![< -];

#[nothing]
a![< -];

#[nothing]
b!{< -}

#[nothing]
c!(< -);

#[nothing]
fn foo() {
    //! dox
    let x = 2 < - 3;
}

fn main() {}
