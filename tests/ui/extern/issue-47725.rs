//@ reference: items.extern.attributes.link_name.allowed-positions

#[link_name = "foo"]
//~^ ERROR attribute cannot be used on
//~| WARN previously accepted
//~| HELP can be applied to
//~| HELP remove the attribute
//~| NOTE `#[deny(harmful_unused_attributes)]` on by default
struct Foo;

#[link_name = "foobar"]
//~^ ERROR attribute cannot be used on
//~| WARN previously accepted
//~| HELP can be applied to
//~| HELP remove the attribute
extern "C" {
    fn foo() -> u32;
}

#[link_name]
//~^ ERROR malformed `link_name` attribute input
//~| HELP must be of the form
//~| ERROR attribute cannot be used on
//~| WARN previously accepted
//~| HELP remove the attribute
//~| HELP can be applied to
//~| NOTE for more information, visit
extern "C" {
    fn bar() -> u32;
}

fn main() {}
