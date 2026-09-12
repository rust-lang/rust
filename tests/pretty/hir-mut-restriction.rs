#![feature(mut_restriction, unsafe_fields)]
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-mut-restriction.pp

struct FooS {
    pub(crate) mut(crate) x: i32,
    mut(self) unsafe y: i32,
}

enum FooE {
    Var {
        mut(crate) unsafe x: i32,
    },
    Tup(mut(self) i32),
}

union FooU {
    pub mut(crate) unsafe x: i32,
    mut(self) y: i32,
}

mod a {
    struct BarS {
        pub(super) mut(self) unsafe x: i32,
        mut(in crate::a) y: i32,
    }
    struct BazS(
        pub(in crate::a) mut(super) i32,
    );

    enum BarE {
        Var {
            mut(super) unsafe x: i32,
        },
        Tup(mut(in crate::a) i32),
    }

    union BarU {
        pub(crate) mut(super) unsafe x: i32,
        mut(in crate::a) y: i32,
    }
}
