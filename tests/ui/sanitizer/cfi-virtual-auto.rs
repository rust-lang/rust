// Tests that calling a trait object method on a trait object with additional auto traits works.

//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C target-feature=-crt-static -C codegen-units=1 -C opt-level=0
//@ run-pass

trait Foo {
    fn foo(&self);
}

struct Bar;
impl Foo for Bar {
    fn foo(&self) {}
}

pub fn main() {
    let x: &(dyn Foo + Send) = &Bar;
    x.foo();
}
