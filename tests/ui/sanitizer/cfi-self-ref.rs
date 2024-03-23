// Check that encoding self-referential types works with #[repr(transparent)]

//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C target-feature=-crt-static -C codegen-units=1 -C opt-level=0
//@ run-pass

use std::marker::PhantomData;

struct X<T> {
    _x: u8,
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
    x.foo(Y(X {_x: 0, p: PhantomData}));
}
