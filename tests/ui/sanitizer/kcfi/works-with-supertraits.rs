// Verifies that super-trait methods can be called through trait objects, and
// that trait objects can be upcast.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    type Output1;
    fn foo(&self) -> Self::Output1;
    // <Type1 as Trait1>::qux is transformed into <dyn Trait1<Output1 = u16> as Trait1>::qux
    fn qux(&self) -> i32 {
        4
    }
}

trait Trait2 {
    type Output2;
    fn bar(&self) -> Self::Output2;
}

trait Trait3: Trait1 + Trait2 {
    type Output3;
    fn baz(&self) -> Self::Output3;
}

struct Type1;

impl Trait1 for Type1 {
    type Output1 = u16;
    // <Type1 as Trait1>::foo is transformed into <dyn Trait1<Output1 = u16> as Trait1>::foo
    fn foo(&self) -> Self::Output1 {
        1
    }
}

impl Trait2 for Type1 {
    type Output2 = u32;
    // <Type1 as Trait2>::bar is transformed into <dyn Trait2<Output2 = u32> as Trait2>::bar
    fn bar(&self) -> Self::Output2 {
        2
    }
}

impl Trait3 for Type1 {
    type Output3 = u8;
    // <Type1 as Trait3>::baz is transformed into
    // <dyn Trait3<Output3 = u8, Output1 = u16, Output2 = u32> as Trait3>::baz.
    fn baz(&self) -> Self::Output3 {
        3
    }
}

fn main() {
    // Methods of a trait and of its supertraits, through a child trait object
    let x = &Type1 as &dyn Trait3<Output3 = u8, Output1 = u16, Output2 = u32>;
    // The virtual method call is transformed into
    // <dyn Trait3<Output3 = u8, Output1 = u16, Output2 = u32> as Trait3>::baz.
    assert_eq!(x.baz(), 3);
    // The virtual method call is transformed into <dyn Trait1<Output1 = u16> as Trait1>::foo
    assert_eq!(x.foo(), 1);
    // The virtual method call is transformed into <dyn Trait2<Output2 = u32> as Trait2>::bar
    assert_eq!(x.bar(), 2);
    // The virtual method call is transformed into <dyn Trait1<Output1 = u16> as Trait1>::qux
    assert_eq!(x.qux(), 4);

    // Methods of a supertrait, through a supertrait object
    let y = &Type1 as &dyn Trait1<Output1 = u16>;
    // The virtual method call is transformed into <dyn Trait1<Output1 = u16> as Trait1>::foo
    assert_eq!(y.foo(), 1);
    // The virtual method call is transformed into <dyn Trait1<Output1 = u16> as Trait1>::qux
    assert_eq!(y.qux(), 4);
    let z = &Type1 as &dyn Trait2<Output2 = u32>;
    // The virtual method call is transformed into <dyn Trait2<Output2 = u32> as Trait2>::bar
    assert_eq!(z.bar(), 2);

    // Methods of a supertrait, through an upcast trait object
    let x1 = x as &dyn Trait1<Output1 = u16>;
    // The virtual method call is transformed into <dyn Trait1<Output1 = u16> as Trait1>::foo
    assert_eq!(x1.foo(), 1);
    // The virtual method call is transformed into <dyn Trait1<Output1 = u16> as Trait1>::qux
    assert_eq!(x1.qux(), 4);
    let x2 = x as &dyn Trait2<Output2 = u32>;
    // The virtual method call is transformed into <dyn Trait2<Output2 = u32> as Trait2>::bar
    assert_eq!(x2.bar(), 2);
}
