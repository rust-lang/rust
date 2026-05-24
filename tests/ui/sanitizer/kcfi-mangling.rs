// Check KCFI extra mangling works correctly on v0

//@ needs-sanitizer-kcfi
//@ no-prefer-dynamic
//@ compile-flags: -Cpanic=abort -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=kcfi -Csymbol-mangling-version=v0
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
