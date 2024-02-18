//@ edition:2018
//@ aux-crate:issue_87932_a=issue-87932-a.rs

pub struct A {}

impl issue_87932_a::Deserialize for A {
    fn deserialize() {
        extern crate issue_87932_a as _a;
    }
}

fn main() {
    A::deserialize();
    //~^ ERROR no function or associated item named `deserialize` found for struct `A`
}
