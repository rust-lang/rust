// Verifies that trait methods (i.e., both trait method implementations in impl
// blocks and provided (default) trait methods in trait blocks) can be called
// through trait objects.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    fn foo(&self) -> i32;
    // <Type1 as Trait1>::bar is transformed into <dyn Trait1 as Trait1>::bar
    fn bar(&self) -> i32 {
        2
    }
    // <Type1 as Trait1>::baz is not transformed, as it can not be called through a vtable
    fn baz<T>(&self) -> i32
    where
        Self: Sized,
    {
        3
    }
}

trait Trait2 {
    fn qux<T>(&self) -> i32;
}

struct Type1;

impl Trait1 for Type1 {
    // <Type1 as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo
    fn foo(&self) -> i32 {
        1
    }
}

impl Trait2 for Type1 {
    // <Type1 as Trait2>::qux is not transformed, as Trait2 is not dyn compatible
    fn qux<T>(&self) -> i32 {
        4
    }
}

fn main() {
    // Trait methods, through a reference to a trait object
    let x = &Type1 as &dyn Trait1;
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::foo
    assert_eq!(x.foo(), 1);
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::bar
    assert_eq!(x.bar(), 2);

    // Trait methods, through a boxed trait object
    let x: Box<dyn Trait1> = Box::new(Type1);
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::foo
    assert_eq!(x.foo(), 1);
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::bar
    assert_eq!(x.bar(), 2);

    // Trait methods that can not be called through a vtable
    // <Type1 as Trait1>::baz is not transformed, as it can not be called through a vtable
    assert_eq!(Type1.baz::<u8>(), 3);
    // <Type1 as Trait2>::qux is not transformed, as Trait2 is not dyn compatible
    assert_eq!(Type1.qux::<u8>(), 4);
}
