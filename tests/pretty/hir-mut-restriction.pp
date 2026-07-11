#![attr = Feature([mut_restriction#0, unsafe_fields#0])]
extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-mut-restriction.pp

struct FooS {
    mut(in crate) x: i32,
    mut(in self) y: i32,
}

enum FooE {
    Var {
            mut(in crate) x: i32,
        },
    Tup(mut(in self) i32),
}

union FooU {
    mut(in crate) x: i32,
    mut(in self) y: i32,
}

mod a {
    struct BarS {
        mut(in self) x: i32,
        mut(in crate::a) y: i32,
    }
    struct BazS(mut(in super) i32);

    enum BarE {
        Var {
                mut(in super) x: i32,
            },
        Tup(mut(in crate::a) i32),
    }

    union BarU {
        mut(in super) x: i32,
        mut(in crate::a) y: i32,
    }
}
