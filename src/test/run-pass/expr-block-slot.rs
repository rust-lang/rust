// Regression test for issue #377


struct A { a: isize }
struct V { v: isize }

pub fn main() {
    let a = { let b = A {a: 3}; b };
    assert_eq!(a.a, 3);
    let c = { let d = V {v: 3}; d };
    assert_eq!(c.v, 3);
}
