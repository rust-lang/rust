//@ compile-flags: -O
//@ run-pass

struct Foo {
    x: i32,
}

pub fn main() {
    let mut foo = Foo { x: 42 };
    let x = &mut foo.x;
    *x = 13;
    let y = foo;
    assert_eq!(y.x, 13); // used to print 42 due to mir-opt bug
}
