// Regression test for https://github.com/rust-lang/rust/issues/150969

#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]

fn pass_enum<const N: usize, const M: usize = const { N }> {
    //~^ ERROR: missing parameters for function definition
    //~| ERROR: defaults for generic parameters are not allowed here
    //~| ERROR: overly complex generic constant
    pass_enum::<{ core::direct_const_arg!(None) }>
    //~^ ERROR: missing generics for enum `Option` [E0107]
}

fn main() {}
