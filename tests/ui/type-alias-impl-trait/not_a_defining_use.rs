#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type Two<T, U> = impl Debug;

#[define_opaque(Two)]
fn three<T: Debug, U>(t: T) -> Two<T, U> {
    (t, 5i8)
}

trait Bar {
    type Blub: Debug;
    const FOO: Self::Blub;
}

impl Bar for u32 {
    type Blub = i32;
    const FOO: i32 = 42;
}

#[define_opaque(Two)]
fn four<T: Debug, U: Bar>(t: T) -> Two<T, U> {
    //~^ ERROR concrete type differs
    (t, <U as Bar>::FOO)
}

fn is_sync<T: Sync>() {}

fn asdfl() {
    //FIXME(oli-obk): these currently cause cycle errors
    //is_sync::<Two<i32, u32>>();
    //is_sync::<Two<i32, *const i32>>();
}
