#[rustc_attribute_should_be_reserved]
//~^ ERROR attribute `rustc_attribute_should_be_reserved` is currently unknown
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

macro_rules! foo {
    () => (());
}

fn main() {
    foo!(); //~ ERROR cannot determine resolution for the macro `foo`
}
