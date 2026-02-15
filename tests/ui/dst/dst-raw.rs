//@ run-pass
// Test DST raw pointers

#![allow(dangerous_implicit_autorefs)]

trait Trait {
    fn foo(&self) -> isize;
}

struct A {
    f: isize
}
impl Trait for A {
    fn foo(&self) -> isize {
        self.f
    }
}

struct Foo<T: ?Sized> {
    f: T
}

pub fn main() {
    // raw trait object
    let x = A { f: 42 };
    let z: *const dyn Trait = &x;
    let r = unsafe {
        (&*z).foo()
    };
    assert_eq!(r, 42);

    // raw DST struct
    let p = Foo {f: A { f: 42 }};
    let o: *const Foo<dyn Trait> = &p;
    let r = unsafe {
        (&*o).f.foo()
    };
    assert_eq!(r, 42);

    // raw slice
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let b = (*a)[2];
        assert_eq!(b, 3);
        let len = (*a).len();
        assert_eq!(len, 3);
    }

    // raw slice with explicit cast
    let a = &[1, 2, 3] as *const [i32];
    unsafe {
        let b = (*a)[2];
        assert_eq!(b, 3);
        let len = (*a).len();
        assert_eq!(len, 3);
    }

    // raw DST struct with slice
    let c: *const Foo<[_]> = &Foo {f: [1, 2, 3]};
    unsafe {
        let b = (&*c).f[0];
        assert_eq!(b, 1);
        let len = (&*c).f.len();
        assert_eq!(len, 3);
    }

    // all of the above with *mut
    let mut x = A { f: 42 };
    let z: *mut dyn Trait = &mut x;
    let r = unsafe {
        (&*z).foo()
    };
    assert_eq!(r, 42);

    let mut p = Foo {f: A { f: 42 }};
    let o: *mut Foo<dyn Trait> = &mut p;
    let r = unsafe {
        (&*o).f.foo()
    };
    assert_eq!(r, 42);

    let a: *mut [_] = &mut [1, 2, 3];
    unsafe {
        let b = (*a)[2];
        assert_eq!(b, 3);
        let len = (*a).len();
        assert_eq!(len, 3);
    }

    let a = &mut [1, 2, 3] as *mut [i32];
    unsafe {
        let b = (*a)[2];
        assert_eq!(b, 3);
        let len = (*a).len();
        assert_eq!(len, 3);
    }

    let c: *mut Foo<[_]> = &mut Foo {f: [1, 2, 3]};
    unsafe {
        let b = (&*c).f[0];
        assert_eq!(b, 1);
        let len = (&*c).f.len();
        assert_eq!(len, 3);
    }
}
