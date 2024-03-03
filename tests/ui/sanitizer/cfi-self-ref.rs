// Check that encoding self-referential types works with #[repr(transparent)]

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ build-pass

use std::marker::PhantomData;

struct X<T> {
    x: u8,
    p: PhantomData<T>,
}

#[repr(transparent)]
struct Y(X<Y>);

trait Fooable {
    fn foo(&self, y: Y);
}

struct Bar;

impl Fooable for Bar {
    fn foo(&self, _: Y) {}
}

fn main() {
    let x = &Bar as &dyn Fooable;
    x.foo(Y(X {x: 0, p: PhantomData}));
}
