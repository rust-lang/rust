#[rustc_attribute_should_be_reserved] //~ ERROR attributes with the prefix `rustc_` are reserved
macro_rules! foo {
    () => (());
}

fn main() {
    foo!();
}
