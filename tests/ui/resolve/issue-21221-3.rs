// testing whether the lookup mechanism picks up types
// defined in the outside crate

// aux-build:issue-21221-3.rs

extern crate issue_21221_3;

struct Foo;

// NOTE: This shows only traits accessible from the current
// crate, thus the two private entities:
//   `issue_21221_3::outer::private_module::OuterTrait` and
//   `issue_21221_3::outer::public_module::OuterTrait`
// are hidden from the view.
impl OuterTrait for Foo {}
//~^ ERROR cannot find trait `OuterTrait`
fn main() {
    println!("Hello, world!");
}
