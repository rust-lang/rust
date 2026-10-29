//@ aux-build: external-mut-restriction.rs
//@ edition: 2018..
#![feature(mut_restriction)]

extern crate external_mut_restriction as external;

pub mod foo {
    pub mod bar {
        #[derive(Default)]
        pub struct FooS {
            pub mut(self) alpha: u8,
            pub mut(super) beta: u8,
            pub mut(crate) gamma: u8,
        }

        pub enum FooE {
            Var {
                mut(in crate::foo::bar) alpha: u8,
                mut(in crate::foo) beta: u8,
                gamma: u8,
            },
            Tup(mut(in crate::foo::bar) u8, mut(in crate::foo) u8, u8),
        }

        pub union FooU {
            pub mut(in self) alpha: u8,
            pub mut(in super) beta: u8,
            pub mut(in super::super) gamma: u8,
        }

        #[derive(Default)]
        pub struct Bar(
            pub mut(self) u8,
            pub mut(super) u8,
            pub mut(crate) u8,
        );

        fn construct_inner() {
            let _ = FooS { alpha: 1, beta: 2, gamma: 3 }; // ok
            let _ = FooE::Var { alpha: 1, beta: 2, gamma: 3 }; // ok
            let _ = FooE::Tup(1, 2, 3); // ok
            let _ = FooU { alpha: 1 }; // ok
            let _ = FooU { beta: 2 }; // ok
            let _ = FooU { gamma: 3 }; // ok
            let _ = Bar(1, 2, 3); // ok
        }
    }

    fn construct_outer() {
        let _ = bar::FooS { alpha: 1, beta: 2, gamma: 3 }; //~ ERROR `FooS` cannot be constructed using a `struct` expression outside `crate::foo::bar`
        let foos = bar::FooS::default();
        let _ = bar::FooS { gamma: 3, ..foos }; //~ ERROR `FooS` cannot be constructed using a `struct` expression outside `crate::foo::bar`

        let _ = bar::FooE::Var { alpha: 1, beta: 2, gamma: 3 }; //~ ERROR `Var` cannot be constructed using a `variant` expression outside `crate::foo::bar`
        let _ = bar::FooE::Tup(1, 2, 3); //~ ERROR `Tup` cannot be constructed using a `variant` expression outside `crate::foo::bar`

        let _ = bar::FooU { alpha: 1 }; //~ ERROR `FooU` cannot be constructed using a `union` expression outside `crate::foo::bar`
        let _ = bar::FooU { beta: 2 }; // ok
        let _ = bar::FooU { gamma: 3 }; // ok

        let _ = bar::Bar(1, 2, 3); //~ ERROR `Bar` cannot be constructed using a `struct` expression outside `crate::foo::bar`
    }
}

fn main() {
    let _ = external::TopLevelS { x: 1, y: 2 }; //~ ERROR `TopLevelS` cannot be constructed using a `struct` expression outside `external`
    let ext_toplevel_s = external::TopLevelS::default();
    let _ = external::TopLevelS { y: 2, ..ext_toplevel_s }; //~ ERROR `TopLevelS` cannot be constructed using a `struct` expression outside `external`

    let _ = external::TopLevelE::Var { x: 1, y: 2 }; //~ ERROR `Var` cannot be constructed using a `variant` expression outside `external`
    let _ = external::TopLevelE::Tup(1, 2); //~ ERROR `Tup` cannot be constructed using a `variant` expression outside `external`
    let _ = external::TopLevelU { x: 1 }; //~ ERROR `TopLevelU` cannot be constructed using a `union` expression outside `external`
    let _ = external::TopLevelU { y: 2 }; // ok
}
