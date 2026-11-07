//@ aux-build: external-mut-restriction.rs
#![feature(mut_restriction)]

extern crate external_mut_restriction as external;

pub mod a {
    pub mod b {
        pub struct Foo {
            pub mut(in a::b) i1: i32, //~ ERROR cannot find module or crate `a` in this scope [E0433]
            pub mut(in ::std) i2: i32, //~ ERROR field mutation can only be restricted to ancestor modules
            pub mut(in self::c) i3: i32, //~ ERROR field mutation can only be restricted to ancestor modules
            pub mut(in super::d) i4: i32, //~ ERROR field mutation can only be restricted to ancestor modules
            pub mut(in crate::c) i5: i32, //~ ERROR cannot find module `c` in the crate root [E0433]
            pub mut(in super::E) i6: i32, //~ ERROR expected module, found enum `super::E` [E0577]
            pub mut(in super::super::super) i7: i32, //~ ERROR too many leading `super` keywords within `crate::a::b` [E0433]

            // OK paths
            pub mut(crate) i8: i32,
            pub mut(self) i9: i32,
            pub mut(super) i10: i32,
            pub mut(in crate::a) i11: i32,
            pub mut(in super::super) i12: i32,
        }
        pub mod c {}
    }
    pub mod d {}
    pub enum E {}
}

pub enum Foo {
    Var {
        mut(in crate::a) e1: i32, //~ ERROR field mutation can only be restricted to ancestor modules
        mut(in crate::a::E) e2: i32, //~ ERROR expected module, found enum `crate::a::E` [E0577]
        mut(crate) e3: i32,
        mut(self) e4: i32,
        mut(super) e5: i32, //~ ERROR too many leading `super` keywords within `crate` [E0433]
    },
    Tup(mut(in external) i32), //~ ERROR field mutation can only be restricted to ancestor modules
}

// Check if we can resolve `use`d paths.
pub mod m1 {
    pub union Foo {
        pub mut(in crate::m2) x: i32, // OK
    }
}
use m1 as m2;

pub mod m3 {
    pub mod m4 {
        pub struct Bar (
            pub mut(in crate::m2) i32, //~ ERROR field mutation can only be restricted to ancestor modules
            pub mut(in m6) i32, // OK
            pub mut(in m6::m5) i32, //~ ERROR field mutation can only be restricted to ancestor modules
            pub mut(in m7) i32, //~ ERROR expected module, found enum `m7` [E0577]
        );
        use crate::m3 as m6;
        use crate::m3::E as m7;
    }
    mod m5 {}
    pub enum E {}
}

fn main() {}
