// Verifies that methods and functions can be cast to function pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn foo1(_: &Type2) -> i32 {
    1
}

trait Trait1 {
    fn foo(&self) -> i32;
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) -> i32 {
        2
    }
}

trait Trait2 {
    fn foo(&self) -> i32;
    fn bar(&self) -> i32;
}

struct Type2;

impl Trait2 for Type2 {
    fn foo(&self) -> i32 {
        3
    }
    #[track_caller]
    fn bar(&self) -> i32 {
        4
    }
}

fn main() {
    // Trait method implementations cast to function pointers
    // The methods below are transformed, but CFI also attaches secondary type ids with the
    // concrete self type to them (i.e., encoded with the USE_CONCRETE_SELF option), which are the
    // ones tested at the indirect calls.
    let f: fn(&Type1) -> i32 = std::hint::black_box(<Type1 as Trait1>::foo);
    assert_eq!(f(&Type1), 2);
    let f: fn(&Type2) -> i32 = std::hint::black_box(<Type2 as Trait2>::foo);
    assert_eq!(f(&Type2), 3);

    // Non-method functions cast to function pointers
    // foo1 is not transformed, as it is not a trait method or a closure-like
    let f: fn(&Type2) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(&Type2), 1);

    // Trait method implementations with #[track_caller] cast to function pointers
    // The ReifyShim for bar is transformed into <dyn Trait2 as Trait2>::bar, as bar is
    // #[track_caller] and is reified.
    let f: fn(&Type2) -> i32 = std::hint::black_box(<Type2 as Trait2>::bar);
    assert_eq!(f(&Type2), 4);

    // Trait method implementations with #[track_caller], through a vtable
    // The ReifyShim for bar in the vtable is transformed into <dyn Trait2 as Trait2>::bar, as bar
    // is #[track_caller] and is reified.
    let x = &Type2 as &dyn Trait2;
    // The virtual method call is transformed into <dyn Trait2 as Trait2>::bar
    assert_eq!(x.bar(), 4);
}
