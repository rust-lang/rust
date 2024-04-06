// Check that more complex receivers work:
// * Arc<dyn Foo> as for custom receivers
// * &dyn Bar<T=Baz> for type constraints

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

use std::sync::Arc;

trait Foo {
    fn foo(self: Arc<Self>);
}

struct FooImpl;

impl Foo for FooImpl {
    fn foo(self: Arc<Self>) {}
}

trait Bar {
    type T;
    fn bar(&self) -> Self::T;
}

struct BarImpl;

impl Bar for BarImpl {
    type T = i32;
    fn bar(&self) -> Self::T { 7 }
}

fn main() {
    let foo: Arc<dyn Foo> = Arc::new(FooImpl);
    foo.foo();

    let bar: &dyn Bar<T=i32> = &BarImpl;
    assert_eq!(bar.bar(), 7);
}
