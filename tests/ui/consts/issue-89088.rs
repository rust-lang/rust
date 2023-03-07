// Regression test for the ICE described in #89088.

// check-pass

#![allow(indirect_structural_match)]
use std::borrow::Cow;

const FOO: &A = &A::Field(Cow::Borrowed("foo"));

#[derive(PartialEq, Eq)]
enum A {
    Field(Cow<'static, str>)
}

fn main() {
    let var = A::Field(Cow::Borrowed("bar"));

    match &var {
        FOO => todo!(),
        _ => todo!()
    }
}
