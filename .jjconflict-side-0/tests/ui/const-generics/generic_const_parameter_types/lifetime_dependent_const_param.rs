#![feature(generic_const_parameter_types, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

fn foo<'a, const N: &'a u32>() {}

fn bar() {
    foo::<'static, { &1_u32 }>();
    //~^ ERROR: anonymous constants with lifetimes in their type are not yet supported
    foo::<'_, { &1_u32 }>();
    //~^ ERROR: anonymous constants with lifetimes in their type are not yet supported
}

fn borrowck<'a, const N: &'static u32, const M: &'a u32>() {
    foo::<'a, M>();
    foo::<'static, M>();
    //~^ ERROR: lifetime may not live long enough
    foo::<'static, N>();
}

fn main() {}
