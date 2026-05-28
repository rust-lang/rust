#![feature(negative_impls)]

trait MyTrait {
    type Foo;
}

impl !MyTrait for u32 {
    type Foo = i32; //~ ERROR negative impls cannot have any items
}

fn main() {}
