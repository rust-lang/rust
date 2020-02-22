// Test parsing for `default` where it doesn't belong.
// Specifically, we are interested in kinds of items or items in certain contexts.

fn main() {}

#[cfg(FALSE)]
mod free_items {
    default extern crate foo; //~ ERROR item cannot be `default`
    default use foo; //~ ERROR item cannot be `default`
    default static foo: u8; //~ ERROR item cannot be `default`
    default const foo: u8; //~ ERROR item cannot be `default`
    default fn foo(); //~ ERROR item cannot be `default`
    default mod foo {} //~ ERROR item cannot be `default`
    default extern "C" {} //~ ERROR item cannot be `default`
    default type foo = u8; //~ ERROR item cannot be `default`
    default enum foo {} //~ ERROR item cannot be `default`
    default struct foo {} //~ ERROR item cannot be `default`
    default union foo {} //~ ERROR item cannot be `default`
    default trait foo {} //~ ERROR item cannot be `default`
    default trait foo = Ord; //~ ERROR item cannot be `default`
    default impl foo {}
    default!();
    default::foo::bar!();
    default macro foo {} //~ ERROR item cannot be `default`
    default macro_rules! foo {} //~ ERROR item cannot be `default`
}
