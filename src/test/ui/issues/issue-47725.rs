#[link_name = "foo"] //~ ERROR attribute should be applied to a foreign function or static
struct Foo; //~ NOTE not a foreign function or static

#[link_name = "foobar"]
//~^ ERROR attribute should be applied to a foreign function or static
//~| HELP try `#[link(name = "foobar")]` instead
extern "C" {
    fn foo() -> u32;
}
//~^^^ NOTE not a foreign function or static

#[link_name]
//~^ ERROR malformed `link_name` attribute input
//~| HELP must be of the form
//~| ERROR attribute should be applied to a foreign function or static
//~| HELP try `#[link(name = "...")]` instead
extern "C" {
    fn bar() -> u32;
}
//~^^^ NOTE not a foreign function or static

fn main() {}
