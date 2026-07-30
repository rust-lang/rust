// Tests that calling a trait object method on a trait object with additional auto traits works.

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -T sanitizer=cfi -Z unstable-options
//@ [kcfi] compile-flags: -T sanitizer=kcfi -Z unstable-options
//@ [kcfi] compile-flags: -C panic=abort -C prefer-dynamic=off
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
