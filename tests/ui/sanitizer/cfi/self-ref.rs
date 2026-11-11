// Check that encoding self-referential types works with #[repr(transparent)]

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -C prefer-dynamic=off
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
