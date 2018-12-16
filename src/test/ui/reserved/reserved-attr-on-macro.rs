#[rustc_attribute_should_be_reserved]
//~^ ERROR unless otherwise specified, attributes with the prefix `rustc_` are reserved
macro_rules! foo {
    () => (());
}

fn main() {
    foo!(); //~ ERROR cannot determine resolution for the macro `foo`
}
