// Verifies that trait methods can be called through trait objects with
// associated types.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    type Output;
    fn foo(&self) -> Self::Output;
}

struct Type1;

impl Trait1 for Type1 {
    type Output = i32;
    // <Type1 as Trait1>::foo is transformed into <dyn Trait1<Output = i32> as Trait1>::foo
    fn foo(&self) -> Self::Output {
        1
    }
}

trait Trait2 {
    type Output<'a>
    where
        Self: Sized;

    fn bar(&self) -> i32;
}

impl Trait2 for () {
    type Output<'a>
        = ()
    where
        Self: Sized;

    // <() as Trait2>::bar is transformed into <dyn Trait2 as Trait2>::bar
    fn bar(&self) -> i32 {
        2
    }
}

fn main() {
    let x: &dyn Trait1<Output = i32> = &Type1;
    // The virtual method call is transformed into <dyn Trait1<Output = i32> as Trait1>::foo
    assert_eq!(x.foo(), 1);
    let x: &dyn Trait2 = &();
    // The virtual method call is transformed into <dyn Trait2 as Trait2>::bar
    assert_eq!(x.bar(), 2);
}
