//@ run-pass

use std::fmt;
use std::mem;

#[repr(packed)]
#[derive(Copy, Clone)]
struct Foo1 {
    bar: u8,
    baz: u64
}

impl PartialEq for Foo1 {
    fn eq(&self, other: &Foo1) -> bool {
        self.bar == other.bar && self.baz == other.baz
    }
}

impl fmt::Debug for Foo1 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bar = self.bar;
        let baz = self.baz;

        f.debug_struct("Foo1")
            .field("bar", &bar)
            .field("baz", &baz)
            .finish()
    }
}

#[repr(packed(2))]
#[derive(Copy, Clone)]
struct Foo2 {
    bar: u8,
    baz: u64
}

impl PartialEq for Foo2 {
    fn eq(&self, other: &Foo2) -> bool {
        self.bar == other.bar && self.baz == other.baz
    }
}

impl fmt::Debug for Foo2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bar = self.bar;
        let baz = self.baz;

        f.debug_struct("Foo2")
            .field("bar", &bar)
            .field("baz", &baz)
            .finish()
    }
}

#[repr(C, packed(4))]
#[derive(Copy, Clone)]
struct Foo4C {
    bar: u8,
    baz: u64
}

impl PartialEq for Foo4C {
    fn eq(&self, other: &Foo4C) -> bool {
        self.bar == other.bar && self.baz == other.baz
    }
}

impl fmt::Debug for Foo4C {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bar = self.bar;
        let baz = self.baz;

        f.debug_struct("Foo4C")
            .field("bar", &bar)
            .field("baz", &baz)
            .finish()
    }
}

pub fn main() {
    let foo1s = [Foo1 { bar: 1, baz: 2 }; 10];

    assert_eq!(mem::align_of::<[Foo1; 10]>(), 1);
    assert_eq!(mem::size_of::<[Foo1; 10]>(), 90);

    for i in 0..10 {
        assert_eq!(foo1s[i], Foo1 { bar: 1, baz: 2});
    }

    for &foo in &foo1s {
        assert_eq!(foo, Foo1 { bar: 1, baz: 2 });
    }

    let foo2s = [Foo2 { bar: 1, baz: 2 }; 10];

    assert_eq!(mem::align_of::<[Foo2; 10]>(), 2);
    assert_eq!(mem::size_of::<[Foo2; 10]>(), 100);

    for i in 0..10 {
        assert_eq!(foo2s[i], Foo2 { bar: 1, baz: 2});
    }

    for &foo in &foo2s {
        assert_eq!(foo, Foo2 { bar: 1, baz: 2 });
    }

    let foo4s = [Foo4C { bar: 1, baz: 2 }; 10];

    assert_eq!(mem::align_of::<[Foo4C; 10]>(), 4);
    assert_eq!(mem::size_of::<[Foo4C; 10]>(), 120);

    for i in 0..10 {
        assert_eq!(foo4s[i], Foo4C { bar: 1, baz: 2});
    }

    for &foo in &foo4s {
        assert_eq!(foo, Foo4C { bar: 1, baz: 2 });
    }
}
