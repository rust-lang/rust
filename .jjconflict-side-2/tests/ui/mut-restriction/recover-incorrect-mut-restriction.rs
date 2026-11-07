#![feature(mut_restriction, unsafe_fields)]

pub mod foo {
    pub struct Foo {
        pub mut(crate::foo) x: i32, //~ ERROR incorrect `mut` restriction
        pub mut(crate::foo) unsafe y: i32, //~ ERROR incorrect `mut` restriction
    }

    pub struct TupFoo(pub mut(crate::foo) i32); //~ ERROR incorrect `mut` restriction

    pub enum EnumFoo {
        Var {
            mut(crate::foo) x: i32, //~ ERROR incorrect `mut` restriction
            mut(crate::foo) unsafe y: i32 //~ ERROR incorrect `mut` restriction
        },
        Tup(mut(crate::foo) i32), //~ ERROR incorrect `mut` restriction
    }

    pub union UnionFoo {
        pub mut(crate::foo) x: i32, //~ ERROR incorrect `mut` restriction
        pub mut(crate::foo) unsafe y: i32, //~ ERROR incorrect `mut` restriction
    }
}

fn main() {}
