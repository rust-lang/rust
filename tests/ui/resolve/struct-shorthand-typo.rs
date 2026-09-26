//! Regression test for https://github.com/rust-lang/rust/issues/163324

const A: usize = 0;
struct S { a: usize }

fn test_basic() {
    S { a }; //~ ERROR cannot find value `a` in this scope
}

const B: usize = 1;
struct Multi { a: usize, b: usize }
fn test_multi() {
    Multi { a: 0, b }; //~ ERROR cannot find value `b` in this scope
}

const C: usize = 2;
struct Fru { c: usize, d: usize }
fn test_fru() {
    let base = Fru { c: 0, d: 0 };
    Fru { c, ..base }; //~ ERROR cannot find value `c` in this scope
}

const D: usize = 3;
enum E {
    StructVariant { d: usize },
}
fn test_enum() {
    E::StructVariant { d }; //~ ERROR cannot find value `d` in this scope
}

struct Local { e: usize }
fn test_local() {
    let E = 4;
    Local { e }; //~ ERROR cannot find value `e` in this scope
}

fn main() {}
