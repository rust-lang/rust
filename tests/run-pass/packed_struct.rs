#![allow(dead_code)]
#![feature(unsize, coerce_unsized)]

#[repr(packed)]
struct S {
    a: i32,
    b: i64,
}

#[repr(packed)]
struct Test1<'a> {
    x: u8,
    other: &'a u32,
}

#[repr(packed)]
struct Test2<'a> {
    x: u8,
    other: &'a Test1<'a>,
}

fn test(t: Test2) {
    let x = *t.other.other;
    assert_eq!(x, 42);
}

fn test_unsizing() {
    #[repr(packed)]
    struct UnalignedPtr<'a, T: ?Sized>
    where T: 'a,
    {
        data: &'a T,
    }

    impl<'a, T, U> std::ops::CoerceUnsized<UnalignedPtr<'a, U>> for UnalignedPtr<'a, T>
    where
        T: std::marker::Unsize<U> + ?Sized,
        U: ?Sized,
    { }

    let arr = [1, 2, 3];
    let arr_unaligned: UnalignedPtr<[i32; 3]> = UnalignedPtr { data: &arr };
    let arr_unaligned: UnalignedPtr<[i32]> = arr_unaligned;
    let _unused = &arr_unaligned; // forcing an allocation, which could also yield "unaligned write"-errors
}

fn test_drop() {
    struct Wrap(u32);
    impl Drop for Wrap {
        fn drop(&mut self) {
            // Do an (aligned) load
            let _test = self.0;
            // For the fun of it, test alignment
            assert_eq!(&self.0 as *const _ as usize % std::mem::align_of::<u32>(), 0);
        }
    }

    #[repr(packed,C)]
    struct Packed<T> {
        f1: u8, // this should move the second field to something not very aligned
        f2: T,
    }

    let p = Packed { f1: 42, f2: Wrap(23) };
    drop(p);
}

fn test_inner_packed() {
    // Even if just the inner struct is packed, accesses to the outer field can get unaligned.
    // Make sure that works.
    #[repr(packed)]
    #[derive(Clone,Copy)]
    struct Inner(u32);

    #[derive(Clone,Copy)]
    struct Outer(u8, Inner);

    let o = Outer(0, Inner(42));
    let _x = o.1;
    let _y = (o.1).0;
    let _o2 = o.clone();
}

fn main() {
    let mut x = S {
        a: 42,
        b: 99,
    };
    let a = x.a;
    let b = x.b;
    assert_eq!(a, 42);
    assert_eq!(b, 99);
    // can't do `assert_eq!(x.a, 42)`, because `assert_eq!` takes a reference
    assert_eq!({x.a}, 42);
    assert_eq!({x.b}, 99);

    x.b = 77;
    assert_eq!({x.b}, 77);

    test(Test2 { x: 0, other: &Test1 { x: 0, other: &42 }});

    test_unsizing();
    test_drop();
    test_inner_packed();
}
