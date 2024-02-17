//@ aux-build:issue-11680.rs

extern crate issue_11680 as other;

fn main() {
    let _b = other::Foo::Bar(1);
    //~^ ERROR: enum `Foo` is private

    let _b = other::test::Foo::Bar(1);
    //~^ ERROR: enum `Foo` is private
}
