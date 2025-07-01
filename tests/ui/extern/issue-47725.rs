#![warn(unused_attributes)] //~ NOTE lint level is defined here

#[link_name = "foo"]
//~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
struct Foo; //~ NOTE not a foreign function or static

#[link_name = "foobar"]
//~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
//~| HELP try `#[link(name = "foobar")]` instead
extern "C" {
    fn foo() -> u32;
}
//~^^^ NOTE not a foreign function or static

#[link_name]
//~^ ERROR malformed `link_name` attribute input
//~| HELP must be of the form
extern "C" {
    fn bar() -> u32;
}

fn main() {}
