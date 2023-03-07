// aux-build:issue-41652-b.rs

extern crate issue_41652_b;

struct S;

impl issue_41652_b::Tr for S {
    fn f() {
        3.f()
        //~^ ERROR can't call method `f` on ambiguous numeric type `{integer}`
    }
}

fn main() {}
