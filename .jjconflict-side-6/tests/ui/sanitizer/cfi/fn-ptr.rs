// Verifies that casting to a function pointer works.

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer
//@ [cfi] compile-flags: -C opt-level=0 -C codegen-units=1 -C lto
//@ [cfi] compile-flags: -C prefer-dynamic=off
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -C prefer-dynamic=off
//@ run-pass

trait Foo {
    fn foo(&self);
    fn bar(&self);
}

struct S;

impl Foo for S {
    fn foo(&self) {}
    #[track_caller]
    fn bar(&self) {}
}

struct S2 {
    f: fn(&S)
}

impl S2 {
    fn foo(&self, s: &S) {
        (self.f)(s)
    }
}

trait Trait1 {
    fn foo(&self);
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {}
}

fn foo<T>(_: &T) {}

fn main() {
    let type1 = Type1 {};
    let f = <Type1 as Trait1>::foo;
    f(&type1);
    // Check again with different optimization barriers
    S2 { f: <S as Foo>::foo }.foo(&S);
    // Check mismatched #[track_caller]
    S2 { f: <S as Foo>::bar }.foo(&S);
    // Check non-method functions
    S2 { f: foo }.foo(&S)
}
