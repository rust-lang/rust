// Check that we only elaborate non-`Self: Sized` associated types when
// erasing the receiver from trait ref.

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

trait Foo {
    type Bar<'a>
    where
        Self: Sized;

    fn test(&self);
}

impl Foo for () {
    type Bar<'a> = ()
    where
        Self: Sized;

    fn test(&self) {}
}

fn test(x: &dyn Foo) {
    x.test();
}

fn main() {
    test(&());
}
