#![feature(negative_impls)]
#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait MyTrait {
    type Foo;
}

default impl !MyTrait for u32 {}
//~^ ERROR negative impls on primitive types must not contain where bounds

fn main() {}
