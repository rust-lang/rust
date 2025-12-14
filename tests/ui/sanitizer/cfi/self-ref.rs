// Check that encoding self-referential types works with #[repr(transparent)]

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ [cfi] compile-flags: -Ccodegen-units=1 -Clto -Cprefer-dynamic=off
//@ [cfi] compile-flags: -Zunstable-options -Csanitize=cfi
//@ [kcfi] compile-flags: -Cpanic=abort -Cprefer-dynamic=off
//@ [kcfi] compile-flags: -Zunstable-options -Csanitize=kcfi
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize
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
