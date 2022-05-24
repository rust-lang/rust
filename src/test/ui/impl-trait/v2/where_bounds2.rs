// check-pass

#![feature(return_position_impl_trait_v2)]

trait MyTrait {}

fn foo<T: MyTrait>(t: T) -> impl MyTrait {
    t
}

fn main() {}
