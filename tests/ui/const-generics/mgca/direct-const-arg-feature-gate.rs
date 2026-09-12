fn foo<const N: usize>(_: [(); core::direct_const_arg!(N)]) {}
//~^ ERROR use of unstable library feature `min_generic_const_args`
//~| ERROR expected expression, found `direct_const_arg!()` constant
//~| ERROR generic parameters may not be used in const operations
fn main() {}
