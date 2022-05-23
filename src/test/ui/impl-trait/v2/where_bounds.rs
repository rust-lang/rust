// check-pass

#![feature(return_position_impl_trait_v2)]

trait MyTrait {}

fn foo<T>(t: T) -> impl MyTrait
where
    T: MyTrait,
{
    t
}

fn main() {}
