#![feature(mut_restriction)]
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-mut-restriction.pp

struct FooS {
    mut(crate) x: i32,
    mut(self) y: i32,
}

enum FooE {
    Var {
        mut(crate) x: i32,
    },
    Tup(mut(self) i32),
}

union FooU {
    mut(crate) x: i32,
    mut(self) y: i32,
}

mod a {
    struct BarS {
        mut(self) x: i32,
        mut(in crate::a) y: i32,
    }
    struct BazS(
        mut(super) i32,
    );

    enum BarE {
        Var {
            mut(super) x: i32,
        },
        Tup(mut(in crate::a) i32),
    }

    union BarU {
        mut(super) x: i32,
        mut(in crate::a) y: i32,
    }
}
