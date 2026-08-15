#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Wrapper<T>(T);

impl Wrapper<i32> {
    type Foo = i32;
}

impl Wrapper<()> {
    type Foo = ();
}

fn main() {
    let _: Wrapper<_>::Foo = (); //~ ERROR multiple applicable items in scope
}
