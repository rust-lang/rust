#![feature(rustc_attrs)]
#![feature(negative_impls)]

#[rustc_auto_trait]
trait MyTrait {}

#[rustc_auto_trait]
unsafe trait MyUnsafeTrait {}

struct ThisImplsTrait;

impl !MyUnsafeTrait for ThisImplsTrait {}


struct ThisImplsUnsafeTrait;

impl !MyTrait for ThisImplsUnsafeTrait {}

fn is_my_trait<T: MyTrait>() {}
fn is_my_unsafe_trait<T: MyUnsafeTrait>() {}

fn main() {
    is_my_trait::<ThisImplsTrait>();
    is_my_trait::<ThisImplsUnsafeTrait>();
    //~^ ERROR `ThisImplsUnsafeTrait: MyTrait` is not satisfied

    is_my_unsafe_trait::<ThisImplsTrait>();
    //~^ ERROR `ThisImplsTrait: MyUnsafeTrait` is not satisfied

    is_my_unsafe_trait::<ThisImplsUnsafeTrait>();
}
