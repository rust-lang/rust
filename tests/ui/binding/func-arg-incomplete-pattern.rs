// run-pass
#![allow(dead_code)]
// Test that we do not leak when the arg pattern must drop part of the
// argument (in this case, the `y` field).

struct Foo {
    x: Box<usize>,
    y: Box<usize>,
}

fn foo(Foo {x, ..}: Foo) -> *const usize {
    let addr: *const usize = &*x;
    addr
}

pub fn main() {
    let obj: Box<_> = Box::new(1);
    let objptr: *const usize = &*obj;
    let f = Foo { x: obj, y: Box::new(2) };
    let xptr = foo(f);
    assert_eq!(objptr, xptr);
}
