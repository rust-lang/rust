#![feature(type_changing_struct_update)]
#![allow(incomplete_features)]

#[derive(Default)]
pub struct Foo<P, T> {
    pub t: T,
    pub v: P
}

impl<P: Default, T: Default> Foo<P, T> {
    pub fn new() -> Self {
        Foo { t: T::default(), v: P::default() }
    }

    pub fn d(t: T) -> Self {
        Foo { t, ..Default::default() }
    }

    pub fn o(t: T) -> Self {
        let d: Foo<_, _> = Foo::new();
        Foo { t, ..d }
    }

    pub fn o2<P2: Default>(t: T, v: P2) -> (Foo<P, T>, Foo<P2, T>) {
        let d = Default::default();
        let foo1 = Foo { t, ..d };
        let foo2 = Foo { v, ..d };
        (foo1, foo2)
    }

    pub fn o3<T2: Default>(t1: T, t2: T2) -> (Foo<P, T>, Foo<P, T2>) {
        let d = Default::default();
        let foo1 = Foo { t: t1, ..d };
        let foo2 = Foo { t: t2, ..d };
                                  //~^ ERROR mismatched types [E0308]
        (foo1, foo2)
    }
}

fn main() {}
