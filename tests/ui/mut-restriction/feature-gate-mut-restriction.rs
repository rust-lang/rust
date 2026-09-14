//@ revisions: with_gate without_gate
//@[with_gate] check-pass

#![cfg_attr(with_gate, feature(mut_restriction))]
#![feature(unsafe_fields)]

pub struct Foo {
    pub mut(crate) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    pub mut(self) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
}

pub struct TupFoo(pub mut(crate) i32, pub mut(self) i32); //[without_gate]~ ERROR `mut` restrictions are experimental
//[without_gate]~^ ERROR `mut` restrictions are experimental

pub enum EnumFoo {
    Var {
        mut(self) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        mut(crate) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    },
    Tup(mut(self) i32, mut(crate) i32), //[without_gate]~ ERROR `mut` restrictions are experimental
    //[without_gate]~^ ERROR `mut` restrictions are experimental
}

pub union UnionFoo {
    pub mut(self) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    pub mut(crate) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
}

pub mod foo {
    pub struct Bar {
        pub mut(super) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        pub mut(in crate::foo) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    }

    pub struct TupBar(pub mut(super) i32, pub mut(in crate::foo) i32); //[without_gate]~ ERROR `mut` restrictions are experimental
    //[without_gate]~^ ERROR `mut` restrictions are experimental

    pub enum EnumBar {
        Var {
            mut(in crate::foo) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
            mut(super) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        },
        Tup(mut(in crate::foo) i32, mut(super) i32), //[without_gate]~ ERROR `mut` restrictions are experimental
        //[without_gate]~^ ERROR `mut` restrictions are experimental
    }

    pub union UnionBar {
        pub mut(in crate::foo) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        pub mut(super) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    }
}

#[cfg(false)]
pub struct Baz {
    pub mut(crate) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    pub mut(self) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
}

#[cfg(false)]
pub struct TupBaz(pub mut(crate) i32, pub mut(self) i32); //[without_gate]~ ERROR `mut` restrictions are experimental
//[without_gate]~^ ERROR `mut` restrictions are experimental

#[cfg(false)]
pub enum EnumBaz {
    Var {
        mut(self) x: i32, //[without_gate]~ ERROR `mut` restrictions
        mut(crate) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    },
    Tup(mut(self) i32, mut(crate) i32), //[without_gate]~ ERROR `mut` restrictions are experimental
    //[without_gate]~^ ERROR `mut` restrictions are experimental
}

#[cfg(false)]
pub union UnionBaz {
    pub mut(self) x: i32, //[without_gate]~ ERROR `mut`
    pub mut(crate) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
}

#[cfg(false)]
pub mod bar {
    pub struct Bar {
        pub mut(super) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        pub mut(in crate::foo) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    }

    pub struct TupBar(pub mut(super) i32, pub mut(in crate::foo) i32); //[without_gate]~ ERROR `mut` restrictions are experimental
    //[without_gate]~^ ERROR `mut` restrictions are experimental

    pub enum EnumBar {
        Var {
            mut(in crate::foo) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
            mut(super) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        },
        Tup(mut(in crate::foo) i32, mut(super) i32), //[without_gate]~ ERROR `mut` restrictions are experimental
        //[without_gate]~^ ERROR `mut` restrictions are experimental
    }

    pub union UnionBar {
        pub mut(in crate::foo) x: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
        pub mut(super) unsafe y: i32, //[without_gate]~ ERROR `mut` restrictions are experimental
    }
}

fn main() {}
