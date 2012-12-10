// xfail-fast
// aux-build:struct_destructuring_cross_crate.rs

extern mod struct_destructuring_cross_crate;

fn main() {
    let x = struct_destructuring_cross_crate::S { x: 1, y: 2 };
    let struct_destructuring_cross_crate::S { x: a, y: b } = x;
    assert a == 1;
    assert b == 2;
}
