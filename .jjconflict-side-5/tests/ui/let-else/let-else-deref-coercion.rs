// Taken from https://github.com/rust-lang/rust/blob/6cc0a764e082d9c0abcf37a768d5889247ba13e2/compiler/rustc_typeck/src/check/_match.rs#L445-L462
//
// We attempt to `let Bar::Present(_) = foo else { ... }` where foo is meant to Deref/DerefMut to
// Bar. This fails, you must add a type annotation like `let _: &mut Bar = _ else { ... }`


use std::ops::{Deref, DerefMut};

struct Foo(Bar);

enum Bar {
    Present(u32),
    Absent,
}
impl Deref for Foo {
    type Target = Bar;
    fn deref(&self) -> &Bar {
        &self.0
    }
}
impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut Bar {
        &mut self.0
    }
}
impl Bar {
    fn bar(&self) -> Option<u32> {
        let Bar::Present(z): &Bar = self else {
            return None;
        };
        return Some(*z);
    }
}
impl Foo {
    // Try without the type annotation
    fn set_bar_unannotated(&mut self, value: u32) {
        let Bar::Present(z) = self else { //~ ERROR mismatched types
            return;
        };
        *z = value;
    }
}

fn main() {
    let mut foo = Foo(Bar::Present(1));
    foo.set_bar_unannotated(54);
    assert_eq!(foo.bar(), Some(54));
    irrefutable::inner();
}

// The original, to show it fails for irrefutable let decls
mod irrefutable {
    use std::ops::{Deref, DerefMut};
    struct Foo(Bar);
    struct Bar(u32);
    impl Deref for Foo {
        type Target = Bar;
        fn deref(&self) -> &Bar {
            &self.0
        }
    }
    impl DerefMut for Foo {
        fn deref_mut(&mut self) -> &mut Bar {
            &mut self.0
        }
    }
    fn foo(x: &mut Foo) {
        let Bar(z) = x; //~ ERROR mismatched types
        *z = 54;
        assert_eq!((x.0).0, 54);
    }
    pub fn inner() {
        foo(&mut Foo(Bar(1)));
    }
}
