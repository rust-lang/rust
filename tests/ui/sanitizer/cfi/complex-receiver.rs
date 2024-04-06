// Check that more complex receivers work:
// * Arc<dyn Foo> as for custom receivers
// * &dyn Bar<T=Baz> for type constraints

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Zunstable-options -Csanitize=cfi
//@ [kcfi] compile-flags: -Zunstable-options -Csanitize=kcfi
//@ [kcfi] compile-flags: -C panic=abort -C prefer-dynamic=off
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
