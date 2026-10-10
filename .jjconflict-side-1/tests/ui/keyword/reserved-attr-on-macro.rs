#[rustc_attribute_should_be_reserved]
//~^ ERROR cannot find attribute `rustc_attribute_should_be_reserved` in this scope
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

macro_rules! foo {
    () => (());
}

fn main() {
    foo!();
}
