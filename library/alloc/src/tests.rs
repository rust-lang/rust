//! Test for `boxed` mod.

use core::any::Any;
use core::clone::Clone;
use core::convert::TryInto;
use core::ops::Deref;

use std::boxed::Box;

#[test]
fn test_owned_clone() {
    let a = Box::new(5);
    let b: Box<i32> = a.clone();
    assert!(a == b);
}

#[derive(Debug, PartialEq, Eq)]
struct Test;

#[test]
fn any_move() {
    let a = Box::new(8) as Box<dyn Any>;
    let b = Box::new(Test) as Box<dyn Any>;

    let a: Box<i32> = a.downcast::<i32>().unwrap();
    assert_eq!(*a, 8);

    let b: Box<Test> = b.downcast::<Test>().unwrap();
    assert_eq!(*b, Test);

    let a = Box::new(8) as Box<dyn Any>;
    let b = Box::new(Test) as Box<dyn Any>;

    assert!(a.downcast::<Box<i32>>().is_err());
    assert!(b.downcast::<Box<Test>>().is_err());
}

#[test]
fn test_show() {
    let a = Box::new(8) as Box<dyn Any>;
    let b = Box::new(Test) as Box<dyn Any>;
    let a_str = format!("{a:?}");
    let b_str = format!("{b:?}");
    assert_eq!(a_str, "Any { .. }");
    assert_eq!(b_str, "Any { .. }");

    static EIGHT: usize = 8;
    static TEST: Test = Test;
    let a = &EIGHT as &dyn Any;
    let b = &TEST as &dyn Any;
    let s = format!("{a:?}");
    assert_eq!(s, "Any { .. }");
    let s = format!("{b:?}");
    assert_eq!(s, "Any { .. }");
}

#[test]
fn deref() {
    fn homura<T: Deref<Target = i32>>(_: T) {}
    homura(Box::new(765));
}

#[test]
fn raw_sized() {
    let x = Box::new(17);
    let p = Box::into_raw(x);
    unsafe {
        assert_eq!(17, *p);
        *p = 19;
        let y = Box::from_raw(p);
        assert_eq!(19, *y);
    }
}

#[test]
fn raw_trait() {
    trait Foo {
        fn get(&self) -> u32;
        fn set(&mut self, value: u32);
    }

    struct Bar(u32);

    impl Foo for Bar {
        fn get(&self) -> u32 {
            self.0
        }

        fn set(&mut self, value: u32) {
            self.0 = value;
        }
    }

    let x: Box<dyn Foo> = Box::new(Bar(17));
    let p = Box::into_raw(x);
    unsafe {
        assert_eq!(17, (*p).get());
        (*p).set(19);
        let y: Box<dyn Foo> = Box::from_raw(p);
        assert_eq!(19, y.get());
    }
}

#[test]
fn f64_slice() {
    let slice: &[f64] = &[-1.0, 0.0, 1.0, f64::INFINITY];
    let boxed: Box<[f64]> = Box::from(slice);
    assert_eq!(&*boxed, slice)
}

#[test]
fn i64_slice() {
    let slice: &[i64] = &[i64::MIN, -2, -1, 0, 1, 2, i64::MAX];
    let boxed: Box<[i64]> = Box::from(slice);
    assert_eq!(&*boxed, slice)
}

#[test]
fn str_slice() {
    let s = "Hello, world!";
    let boxed: Box<str> = Box::from(s);
    assert_eq!(&*boxed, s)
}

#[test]
fn boxed_slice_from_iter() {
    let iter = 0..100;
    let boxed: Box<[u32]> = iter.collect();
    assert_eq!(boxed.len(), 100);
    assert_eq!(boxed[7], 7);
}

#[test]
fn test_array_from_slice() {
    let v = vec![1, 2, 3];
    let r: Box<[u32]> = v.into_boxed_slice();

    let a: Result<Box<[u32; 3]>, _> = r.clone().try_into();
    assert!(a.is_ok());

    let a: Result<Box<[u32; 2]>, _> = r.clone().try_into();
    assert!(a.is_err());
}
