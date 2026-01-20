#![feature(min_generic_const_args, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const ASSOC: usize;
}

fn takes_tuple<const A: (u32, u32)>() {}
fn takes_nested_tuple<const A: (u32, (u32, u32))>() {}

fn generic_caller<T: Trait, const N: u32, const N2: u32>() {
    takes_tuple::<{ (N, N + 1) }>(); //~ ERROR complex const arguments must be placed inside of a `const` block
    takes_tuple::<{ (N, T::ASSOC + 1) }>(); //~ ERROR complex const arguments must be placed inside of a `const` block

    takes_nested_tuple::<{ (N, (N, N + 1)) }>(); //~ ERROR complex const arguments must be placed inside of a `const` block
    takes_nested_tuple::<{ (N, (N, const { N + 1 })) }>(); //~ ERROR generic parameters may not be used in const operations
}

fn main() {}
