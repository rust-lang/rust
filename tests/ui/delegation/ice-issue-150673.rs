#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn foo<T>() -> T {
        unimplemented!()
    }
}

struct S<T>(T);

trait Trait {
    reuse to_reuse::foo;
}

impl Trait for S {
//~^ ERROR: missing generics for struct `S`
    reuse Trait::foo;
}

fn main() {}
