fn foo<const N: usize>(_: [(); std::gca!(N)]) {}
//~^ ERROR use of unstable library feature `gca_min_const_items`
//~| ERROR expected expression, found `gca!()` constant
//~| ERROR generic parameters may not be used in const operations
fn main() {}
