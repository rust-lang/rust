fn foo<const N: usize>(_: [(); std::gca!(N)]) {}
//~^ ERROR use of unstable library feature `min_generic_const_args`
//~| ERROR expected expression, found `gca!()` constant
//~| ERROR generic parameters may not be used in const operations
fn main() {}
