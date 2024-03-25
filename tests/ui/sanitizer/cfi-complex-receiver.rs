// Check that more complex receivers work:
// * Arc<dyn Foo> as for custom receivers
// * &dyn Bar<T=Baz> for type constraints

//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C target-feature=-crt-static -C codegen-units=1 -C opt-level=0
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
