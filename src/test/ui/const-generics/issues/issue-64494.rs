// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

trait Foo {
    const VAL: usize;
}

trait MyTrait {}

trait True {}
struct Is<const T: bool>;
impl True for Is<{true}> {}

impl<T: Foo> MyTrait for T where Is<{T::VAL == 5}>: True {}
//[full]~^ ERROR constant expression depends on a generic parameter
//[min]~^^ ERROR generic parameters must not be used inside of non trivial constant values
impl<T: Foo> MyTrait for T where Is<{T::VAL == 6}>: True {}
//[full]~^ ERROR constant expression depends on a generic parameter
//[min]~^^ ERROR generic parameters must not be used inside of non trivial constant values
//[min]~| ERROR conflicting implementations of trait `MyTrait`

fn main() {}
