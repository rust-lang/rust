#![warn(clippy::inline_trait_bounds)]

// Free functions

fn inline_simple<T: Clone>() {}
//~^ inline_trait_bounds

fn inline_multiple<T: Clone + Copy, U: core::fmt::Debug>() {}
//~^ inline_trait_bounds

fn inline_lifetime<'a: 'b, 'b>(x: &'a str, y: &'b str) -> &'b str {
    //~^ inline_trait_bounds
    y
}

#[allow(clippy::multiple_bound_locations)]
fn inline_with_where<T: Clone>()
//~^ inline_trait_bounds
where
    T: core::fmt::Debug,
{
}

fn inline_with_const<T: Clone, const N: usize>() {}
//~^ inline_trait_bounds

fn inline_with_return<T: Clone>(val: T) -> T {
    //~^ inline_trait_bounds
    val
}

//  Trait methods

trait MyTrait {
    fn trait_method_inline<T: Clone>(&self);
    //~^ inline_trait_bounds

    fn trait_method_default<T: Clone + Copy>(&self) {}
    //~^ inline_trait_bounds

    fn trait_method_where<T>(&self)
    where
        T: Clone;
}

//   Impl methods

struct MyStruct;

impl MyStruct {
    fn impl_method_inline<T: Clone>(&self) {}
    //~^ inline_trait_bounds

    fn impl_method_multiple<T: Clone, U: core::fmt::Debug>(&self) {}
    //~^ inline_trait_bounds
}

impl MyTrait for MyStruct {
    fn trait_method_inline<T: Clone>(&self) {}
    //~^ inline_trait_bounds

    fn trait_method_default<T: Clone + Copy>(&self) {}
    //~^ inline_trait_bounds

    fn trait_method_where<T>(&self)
    where
        T: Clone,
    {
    }
}

//  Should NOT lint

fn where_only<T>()
where
    T: Clone,
{
}

fn no_bounds<T, U>() {}

fn no_generics() {}

struct InlineStruct<T: Clone>(T);

enum InlineEnum<T: Clone> {
    A(T),
}

#[allow(invalid_type_param_default)]
//~v inline_trait_bounds
fn with_default_value<T: Clone = u32>(x: T) -> T {
    x
}
