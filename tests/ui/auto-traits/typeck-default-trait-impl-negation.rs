#![feature(auto_traits)]
#![feature(negative_impls)]

auto trait MyTrait {}

unsafe auto trait MyUnsafeTrait {}

struct ThisImplsTrait;

impl !MyUnsafeTrait for ThisImplsTrait {}


struct ThisImplsUnsafeTrait;

impl !MyTrait for ThisImplsUnsafeTrait {}

fn is_my_trait<T: MyTrait>() {}
fn is_my_unsafe_trait<T: MyUnsafeTrait>() {}

fn main() {
    is_my_trait::<ThisImplsTrait>();
    is_my_trait::<ThisImplsUnsafeTrait>();
    //~^ ERROR trait `MyTrait` is not implemented for `ThisImplsUnsafeTrait`

    is_my_unsafe_trait::<ThisImplsTrait>();
    //~^ ERROR trait `MyUnsafeTrait` is not implemented for `ThisImplsTrait`

    is_my_unsafe_trait::<ThisImplsUnsafeTrait>();
}
