#[rustc_attribute_should_be_reserved]
//~^ ERROR attribute `rustc_attribute_should_be_reserved` is currently unknown
macro_rules! foo {
    () => (());
}

fn main() {
    foo!(); //~ ERROR cannot determine resolution for the macro `foo`
}
