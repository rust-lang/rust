// Check that we only elaborate non-`Self: Sized` associated types when
// erasing the receiver from trait ref.

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
