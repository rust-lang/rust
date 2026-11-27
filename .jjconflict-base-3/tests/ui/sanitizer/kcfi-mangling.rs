// Check KCFI extra mangling works correctly on v0

//@ needs-sanitizer-kcfi
//@ no-prefer-dynamic
//@ compile-flags: -C panic=abort -Zsanitizer=kcfi -C symbol-mangling-version=v0 -C unsafe-allow-abi-mismatch=sanitizer
//@ build-pass
//@ ignore-backends: gcc

trait Foo {
    fn foo(&self);
}

struct Bar;
impl Foo for Bar {
    fn foo(&self) {}
}

struct Baz;
impl Foo for Baz {
    #[track_caller]
    fn foo(&self) {}
}

fn main() {
    // Produces `ReifyShim(_, ReifyReason::FnPtr)`
    let f: fn(&Bar) = Bar::foo;
    f(&Bar);
    // Produces `ReifyShim(_, ReifyReason::Vtable)`
    let v: &dyn Foo = &Baz as _;
    v.foo();
}
