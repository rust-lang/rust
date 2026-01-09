#![feature(min_generic_const_args, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const ASSOC: usize;
}

fn takes_tuple<const A: (u32, u32)>() {}
fn takes_nested_tuple<const A: (u32, (u32, u32))>() {}

fn generic_caller<T: Trait, const N: usize, const N2: u32>() {
    takes_tuple::<{ (N, N2) }>();
    //~^ ERROR the constant `N` is not of type `u32`
    takes_tuple::<{ (N, T::ASSOC) }>();
    //~^ ERROR the constant `N` is not of type `u32`
    //~| ERROR the constant `<T as Trait>::ASSOC` is not of type `u32`

    takes_nested_tuple::<{ (N, (N, N2)) }>();
    //~^ ERROR the constant `N` is not of type `u32`
    takes_nested_tuple::<{ (N, (N, T::ASSOC)) }>();
    //~^ ERROR the constant `N` is not of type `u32`
    //~| ERROR the constant `<T as Trait>::ASSOC` is not of type `u32`
}

fn main() {}
