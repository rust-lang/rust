// run-pass
// Test structs with always-unsized fields.

#![allow(warnings)]
#![feature(unsize, ptr_metadata)]

use std::mem;
use std::ptr;
use std::slice;

struct Foo<T> {
    f: [T],
}

struct Bar {
    f1: usize,
    f2: [usize],
}

struct Baz {
    f1: usize,
    f2: str,
}

trait Tr {
    fn foo(&self) -> usize;
}

struct St {
    f: usize,
}

impl Tr for St {
    fn foo(&self) -> usize {
        self.f
    }
}

struct Qux<'a> {
    f: Tr + 'a,
}

pub fn main() {
    let _: &Foo<f64>;
    let _: &Bar;
    let _: &Baz;

    let _: Box<Foo<i32>>;
    let _: Box<Bar>;
    let _: Box<Baz>;

    let _ = mem::size_of::<Box<Foo<u8>>>();
    let _ = mem::size_of::<Box<Bar>>();
    let _ = mem::size_of::<Box<Baz>>();

    unsafe {
        struct Foo_<T> {
            f: [T; 3],
        }

        let data: Box<Foo_<i32>> = Box::new(Foo_ { f: [1, 2, 3] });
        let x: &Foo<i32> = mem::transmute(slice::from_raw_parts(&*data, 3));
        assert_eq!(x.f.len(), 3);
        assert_eq!(x.f[0], 1);

        struct Baz_ {
            f1: usize,
            f2: [u8; 5],
        }

        let data: Box<_> =
            Box::new(Baz_ { f1: 42, f2: ['a' as u8, 'b' as u8, 'c' as u8, 'd' as u8, 'e' as u8] });
        let x: &Baz = mem::transmute(slice::from_raw_parts(&*data, 5));
        assert_eq!(x.f1, 42);
        let chs: Vec<char> = x.f2.chars().collect();
        assert_eq!(chs.len(), 5);
        assert_eq!(chs[0], 'a');
        assert_eq!(chs[1], 'b');
        assert_eq!(chs[2], 'c');
        assert_eq!(chs[3], 'd');
        assert_eq!(chs[4], 'e');

        struct Qux_ {
            f: St,
        }

        let obj: Box<St> = Box::new(St { f: 42 });
        let obj: &Tr = &*obj;
        let data: Box<_> = Box::new(Qux_ { f: St { f: 234 } });
        let x: &Qux = &*ptr::from_raw_parts::<Qux>((&*data as *const _).cast(), ptr::metadata(obj));
        assert_eq!(x.f.foo(), 234);
    }
}
