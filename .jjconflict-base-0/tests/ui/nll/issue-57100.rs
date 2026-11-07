#![allow(unused)]


// This tests the error messages for borrows of union fields when the unions are embedded in other
// structs or unions.

#[derive(Clone, Copy, Default)]
struct Leaf {
    l1_u8: u8,
    l2_u8: u8,
}

#[derive(Clone, Copy)]
union First {
    f1_leaf: Leaf,
    f2_leaf: Leaf,
    f3_union: Second,
}

#[derive(Clone, Copy)]
union Second {
    s1_leaf: Leaf,
    s2_leaf: Leaf,
}

struct Root {
    r1_u8: u8,
    r2_union: First,
}

// Borrow a different field of the nested union.
fn nested_union() {
    unsafe {
        let mut r = Root {
            r1_u8: 3,
            r2_union: First { f3_union: Second { s2_leaf: Leaf { l1_u8: 8, l2_u8: 4 } } }
        };

        let mref = &mut r.r2_union.f3_union.s1_leaf.l1_u8;
        //                                  ^^^^^^^
        *mref = 22;
        let nref = &r.r2_union.f3_union.s2_leaf.l1_u8;
        //                              ^^^^^^^
        //~^^ ERROR cannot borrow `r.r2_union.f3_union` (via `r.r2_union.f3_union.s2_leaf.l1_u8`) as immutable because it is also borrowed as mutable (via `r.r2_union.f3_union.s1_leaf.l1_u8`) [E0502]
        println!("{} {}", mref, nref)
    }
}

// Borrow a different field of the first union.
fn first_union() {
    unsafe {
        let mut r = Root {
            r1_u8: 3,
            r2_union: First { f3_union: Second { s2_leaf: Leaf { l1_u8: 8, l2_u8: 4 } } }
        };

        let mref = &mut r.r2_union.f2_leaf.l1_u8;
        //                         ^^^^^^^
        *mref = 22;
        let nref = &r.r2_union.f1_leaf.l1_u8;
        //                     ^^^^^^^
        //~^^ ERROR cannot borrow `r.r2_union` (via `r.r2_union.f1_leaf.l1_u8`) as immutable because it is also borrowed as mutable (via `r.r2_union.f2_leaf.l1_u8`) [E0502]
        println!("{} {}", mref, nref)
    }
}

fn main() {}
