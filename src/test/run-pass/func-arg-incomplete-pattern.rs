// Test that we do not leak when the arg pattern must drop part of the
// argument (in this case, the `y` field).

struct Foo {
    x: ~uint,
    y: ~uint,
}

fn foo(Foo {x, _}: Foo) -> *uint {
    let addr: *uint = &*x;
    addr
}

pub fn main() {
    let obj = ~1;
    let objptr: *uint = &*obj;
    let f = Foo {x: obj, y: ~2};
    let xptr = foo(f);
    assert_eq!(objptr, xptr);
}
