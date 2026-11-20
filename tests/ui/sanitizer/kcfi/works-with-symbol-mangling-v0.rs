// Verifies that KCFI works with the v0 symbol mangling version (i.e., that the
// KCFI extra mangling works correctly on v0).
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Csymbol-mangling-version=v0 -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    fn foo(&self);
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {}
}

struct Type2;

impl Trait1 for Type2 {
    #[track_caller]
    fn foo(&self) {}
}

fn main() {
    // The ReifyShim for foo is not transformed, as it is created with ReifyReason::FnPtr
    let f: fn(&Type1) = std::hint::black_box(Type1::foo);
    f(&Type1);
    // The ReifyShim for foo in the vtable is transformed into <dyn Trait1 as Trait1>::foo, as
    // it is created with ReifyReason::Vtable.
    let x = &Type2 as &dyn Trait1;
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::foo
    x.foo();
}
