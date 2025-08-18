#![warn(unused_attributes)] //~ NOTE lint level is defined here

#[link_name = "foo"]
//~^ WARN attribute cannot be used on
//~| WARN previously accepted
//~| HELP can be applied to
struct Foo;

#[link_name = "foobar"]
//~^ WARN attribute cannot be used on
//~| WARN previously accepted
//~| HELP can be applied to
extern "C" {
    fn foo() -> u32;
}

#[link_name]
//~^ ERROR malformed `link_name` attribute input
//~| HELP must be of the form
//~| WARN attribute cannot be used on
//~| WARN previously accepted
//~| HELP can be applied to
//~| NOTE for more information, visit
extern "C" {
    fn bar() -> u32;
}

fn main() {}
