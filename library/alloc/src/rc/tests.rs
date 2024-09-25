use std::cell::RefCell;
use std::clone::Clone;

use super::*;

#[test]
fn test_clone() {
    let x = Rc::new(RefCell::new(5));
    let y = x.clone();
    *x.borrow_mut() = 20;
    assert_eq!(*y.borrow(), 20);
}

#[test]
fn test_simple() {
    let x = Rc::new(5);
    assert_eq!(*x, 5);
}

#[test]
fn test_simple_clone() {
    let x = Rc::new(5);
    let y = x.clone();
    assert_eq!(*x, 5);
    assert_eq!(*y, 5);
}

#[test]
fn test_destructor() {
    let x: Rc<Box<_>> = Rc::new(Box::new(5));
    assert_eq!(**x, 5);
}

#[test]
fn test_live() {
    let x = Rc::new(5);
    let y = Rc::downgrade(&x);
    assert!(y.upgrade().is_some());
}

#[test]
fn test_dead() {
    let x = Rc::new(5);
    let y = Rc::downgrade(&x);
    drop(x);
    assert!(y.upgrade().is_none());
}

#[test]
fn weak_self_cyclic() {
    struct Cycle {
        x: RefCell<Option<Weak<Cycle>>>,
    }

    let a = Rc::new(Cycle { x: RefCell::new(None) });
    let b = Rc::downgrade(&a.clone());
    *a.x.borrow_mut() = Some(b);

    // hopefully we don't double-free (or leak)...
}

#[test]
fn is_unique() {
    let x = Rc::new(3);
    assert!(Rc::is_unique(&x));
    let y = x.clone();
    assert!(!Rc::is_unique(&x));
    drop(y);
    assert!(Rc::is_unique(&x));
    let w = Rc::downgrade(&x);
    assert!(!Rc::is_unique(&x));
    drop(w);
    assert!(Rc::is_unique(&x));
}

#[test]
fn test_strong_count() {
    let a = Rc::new(0);
    assert!(Rc::strong_count(&a) == 1);
    let w = Rc::downgrade(&a);
    assert!(Rc::strong_count(&a) == 1);
    let b = w.upgrade().expect("upgrade of live rc failed");
    assert!(Rc::strong_count(&b) == 2);
    assert!(Rc::strong_count(&a) == 2);
    drop(w);
    drop(a);
    assert!(Rc::strong_count(&b) == 1);
    let c = b.clone();
    assert!(Rc::strong_count(&b) == 2);
    assert!(Rc::strong_count(&c) == 2);
}

#[test]
fn test_weak_count() {
    let a = Rc::new(0);
    assert!(Rc::strong_count(&a) == 1);
    assert!(Rc::weak_count(&a) == 0);
    let w = Rc::downgrade(&a);
    assert!(Rc::strong_count(&a) == 1);
    assert!(Rc::weak_count(&a) == 1);
    drop(w);
    assert!(Rc::strong_count(&a) == 1);
    assert!(Rc::weak_count(&a) == 0);
    let c = a.clone();
    assert!(Rc::strong_count(&a) == 2);
    assert!(Rc::weak_count(&a) == 0);
    drop(c);
}

#[test]
fn weak_counts() {
    assert_eq!(Weak::weak_count(&Weak::<u64>::new()), 0);
    assert_eq!(Weak::strong_count(&Weak::<u64>::new()), 0);

    let a = Rc::new(0);
    let w = Rc::downgrade(&a);
    assert_eq!(Weak::strong_count(&w), 1);
    assert_eq!(Weak::weak_count(&w), 1);
    let w2 = w.clone();
    assert_eq!(Weak::strong_count(&w), 1);
    assert_eq!(Weak::weak_count(&w), 2);
    assert_eq!(Weak::strong_count(&w2), 1);
    assert_eq!(Weak::weak_count(&w2), 2);
    drop(w);
    assert_eq!(Weak::strong_count(&w2), 1);
    assert_eq!(Weak::weak_count(&w2), 1);
    let a2 = a.clone();
    assert_eq!(Weak::strong_count(&w2), 2);
    assert_eq!(Weak::weak_count(&w2), 1);
    drop(a2);
    drop(a);
    assert_eq!(Weak::strong_count(&w2), 0);
    assert_eq!(Weak::weak_count(&w2), 0);
    drop(w2);
}

#[test]
fn try_unwrap() {
    let x = Rc::new(3);
    assert_eq!(Rc::try_unwrap(x), Ok(3));
    let x = Rc::new(4);
    let _y = x.clone();
    assert_eq!(Rc::try_unwrap(x), Err(Rc::new(4)));
    let x = Rc::new(5);
    let _w = Rc::downgrade(&x);
    assert_eq!(Rc::try_unwrap(x), Ok(5));
}

#[test]
fn into_inner() {
    let x = Rc::new(3);
    assert_eq!(Rc::into_inner(x), Some(3));

    let x = Rc::new(4);
    let y = Rc::clone(&x);
    assert_eq!(Rc::into_inner(x), None);
    assert_eq!(Rc::into_inner(y), Some(4));

    let x = Rc::new(5);
    let _w = Rc::downgrade(&x);
    assert_eq!(Rc::into_inner(x), Some(5));
}

#[test]
fn into_from_raw() {
    let x = Rc::new(Box::new("hello"));
    let y = x.clone();

    let x_ptr = Rc::into_raw(x);
    drop(y);
    unsafe {
        assert_eq!(**x_ptr, "hello");

        let x = Rc::from_raw(x_ptr);
        assert_eq!(**x, "hello");

        assert_eq!(Rc::try_unwrap(x).map(|x| *x), Ok("hello"));
    }
}

#[test]
fn test_into_from_raw_unsized() {
    use std::fmt::Display;
    use std::string::ToString;

    let rc: Rc<str> = Rc::from("foo");

    let ptr = Rc::into_raw(rc.clone());
    let rc2 = unsafe { Rc::from_raw(ptr) };

    assert_eq!(unsafe { &*ptr }, "foo");
    assert_eq!(rc, rc2);

    let rc: Rc<dyn Display> = Rc::new(123);

    let ptr = Rc::into_raw(rc.clone());
    let rc2 = unsafe { Rc::from_raw(ptr) };

    assert_eq!(unsafe { &*ptr }.to_string(), "123");
    assert_eq!(rc2.to_string(), "123");
}

#[test]
fn into_from_weak_raw() {
    let x = Rc::new(Box::new("hello"));
    let y = Rc::downgrade(&x);

    let y_ptr = Weak::into_raw(y);
    unsafe {
        assert_eq!(**y_ptr, "hello");

        let y = Weak::from_raw(y_ptr);
        let y_up = Weak::upgrade(&y).unwrap();
        assert_eq!(**y_up, "hello");
        drop(y_up);

        assert_eq!(Rc::try_unwrap(x).map(|x| *x), Ok("hello"));
    }
}

#[test]
fn test_into_from_weak_raw_unsized() {
    use std::fmt::Display;
    use std::string::ToString;

    let arc: Rc<str> = Rc::from("foo");
    let weak: Weak<str> = Rc::downgrade(&arc);

    let ptr = Weak::into_raw(weak.clone());
    let weak2 = unsafe { Weak::from_raw(ptr) };

    assert_eq!(unsafe { &*ptr }, "foo");
    assert!(weak.ptr_eq(&weak2));

    let arc: Rc<dyn Display> = Rc::new(123);
    let weak: Weak<dyn Display> = Rc::downgrade(&arc);

    let ptr = Weak::into_raw(weak.clone());
    let weak2 = unsafe { Weak::from_raw(ptr) };

    assert_eq!(unsafe { &*ptr }.to_string(), "123");
    assert!(weak.ptr_eq(&weak2));
}

#[test]
fn get_mut() {
    let mut x = Rc::new(3);
    *Rc::get_mut(&mut x).unwrap() = 4;
    assert_eq!(*x, 4);
    let y = x.clone();
    assert!(Rc::get_mut(&mut x).is_none());
    drop(y);
    assert!(Rc::get_mut(&mut x).is_some());
    let _w = Rc::downgrade(&x);
    assert!(Rc::get_mut(&mut x).is_none());
}

#[test]
fn test_cowrc_clone_make_unique() {
    let mut cow0 = Rc::new(75);
    let mut cow1 = cow0.clone();
    let mut cow2 = cow1.clone();

    assert!(75 == *Rc::make_mut(&mut cow0));
    assert!(75 == *Rc::make_mut(&mut cow1));
    assert!(75 == *Rc::make_mut(&mut cow2));

    *Rc::make_mut(&mut cow0) += 1;
    *Rc::make_mut(&mut cow1) += 2;
    *Rc::make_mut(&mut cow2) += 3;

    assert!(76 == *cow0);
    assert!(77 == *cow1);
    assert!(78 == *cow2);

    // none should point to the same backing memory
    assert!(*cow0 != *cow1);
    assert!(*cow0 != *cow2);
    assert!(*cow1 != *cow2);
}

#[test]
fn test_cowrc_clone_unique2() {
    let mut cow0 = Rc::new(75);
    let cow1 = cow0.clone();
    let cow2 = cow1.clone();

    assert!(75 == *cow0);
    assert!(75 == *cow1);
    assert!(75 == *cow2);

    *Rc::make_mut(&mut cow0) += 1;

    assert!(76 == *cow0);
    assert!(75 == *cow1);
    assert!(75 == *cow2);

    // cow1 and cow2 should share the same contents
    // cow0 should have a unique reference
    assert!(*cow0 != *cow1);
    assert!(*cow0 != *cow2);
    assert!(*cow1 == *cow2);
}

#[test]
fn test_cowrc_clone_weak() {
    let mut cow0 = Rc::new(75);
    let cow1_weak = Rc::downgrade(&cow0);

    assert!(75 == *cow0);
    assert!(75 == *cow1_weak.upgrade().unwrap());

    *Rc::make_mut(&mut cow0) += 1;

    assert!(76 == *cow0);
    assert!(cow1_weak.upgrade().is_none());
}

/// This is similar to the doc-test for `Rc::make_mut()`, but on an unsized type (slice).
#[test]
fn test_cowrc_unsized() {
    use std::rc::Rc;

    let mut data: Rc<[i32]> = Rc::new([10, 20, 30]);

    Rc::make_mut(&mut data)[0] += 1; // Won't clone anything
    let mut other_data = Rc::clone(&data); // Won't clone inner data
    Rc::make_mut(&mut data)[1] += 1; // Clones inner data
    Rc::make_mut(&mut data)[2] += 1; // Won't clone anything
    Rc::make_mut(&mut other_data)[0] *= 10; // Won't clone anything

    // Now `data` and `other_data` point to different allocations.
    assert_eq!(*data, [11, 21, 31]);
    assert_eq!(*other_data, [110, 20, 30]);
}

#[test]
fn test_show() {
    let foo = Rc::new(75);
    assert_eq!(format!("{foo:?}"), "75");
}

#[test]
fn test_unsized() {
    let foo: Rc<[i32]> = Rc::new([1, 2, 3]);
    assert_eq!(foo, foo.clone());
}

#[test]
fn test_maybe_thin_unsized() {
    // If/when custom thin DSTs exist, this test should be updated to use one
    use std::ffi::{CStr, CString};

    let x: Rc<CStr> = Rc::from(CString::new("swordfish").unwrap().into_boxed_c_str());
    assert_eq!(format!("{x:?}"), "\"swordfish\"");
    let y: Weak<CStr> = Rc::downgrade(&x);
    drop(x);

    // At this point, the weak points to a dropped DST
    assert!(y.upgrade().is_none());
    // But we still need to be able to get the alloc layout to drop.
    // CStr has no drop glue, but custom DSTs might, and need to work.
    drop(y);
}

#[test]
fn test_from_owned() {
    let foo = 123;
    let foo_rc = Rc::from(foo);
    assert!(123 == *foo_rc);
}

#[test]
fn test_new_weak() {
    let foo: Weak<usize> = Weak::new();
    assert!(foo.upgrade().is_none());
}

#[test]
fn test_ptr_eq() {
    let five = Rc::new(5);
    let same_five = five.clone();
    let other_five = Rc::new(5);

    assert!(Rc::ptr_eq(&five, &same_five));
    assert!(!Rc::ptr_eq(&five, &other_five));
}

#[test]
fn test_from_str() {
    let r: Rc<str> = Rc::from("foo");

    assert_eq!(&r[..], "foo");
}

#[test]
fn test_copy_from_slice() {
    let s: &[u32] = &[1, 2, 3];
    let r: Rc<[u32]> = Rc::from(s);

    assert_eq!(&r[..], [1, 2, 3]);
}

#[test]
fn test_clone_from_slice() {
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct X(u32);

    let s: &[X] = &[X(1), X(2), X(3)];
    let r: Rc<[X]> = Rc::from(s);

    assert_eq!(&r[..], s);
}

#[test]
#[should_panic]
fn test_clone_from_slice_panic() {
    use std::string::{String, ToString};

    struct Fail(u32, String);

    impl Clone for Fail {
        fn clone(&self) -> Fail {
            if self.0 == 2 {
                panic!();
            }
            Fail(self.0, self.1.clone())
        }
    }

    let s: &[Fail] =
        &[Fail(0, "foo".to_string()), Fail(1, "bar".to_string()), Fail(2, "baz".to_string())];

    // Should panic, but not cause memory corruption
    let _r: Rc<[Fail]> = Rc::from(s);
}

#[test]
fn test_from_box() {
    let b: Box<u32> = Box::new(123);
    let r: Rc<u32> = Rc::from(b);

    assert_eq!(*r, 123);
}

#[test]
fn test_from_box_str() {
    use std::string::String;

    let s = String::from("foo").into_boxed_str();
    assert_eq!((&&&s).as_str(), "foo");

    let r: Rc<str> = Rc::from(s);
    assert_eq!((&r).as_str(), "foo");
    assert_eq!(r.as_str(), "foo");

    assert_eq!(&r[..], "foo");
}

#[test]
fn test_from_box_slice() {
    let s = vec![1, 2, 3].into_boxed_slice();
    let r: Rc<[u32]> = Rc::from(s);

    assert_eq!(&r[..], [1, 2, 3]);
}

#[test]
fn test_from_box_trait() {
    use std::fmt::Display;
    use std::string::ToString;

    let b: Box<dyn Display> = Box::new(123);
    let r: Rc<dyn Display> = Rc::from(b);

    assert_eq!(r.to_string(), "123");
}

#[test]
fn test_from_box_trait_zero_sized() {
    use std::fmt::Debug;

    let b: Box<dyn Debug> = Box::new(());
    let r: Rc<dyn Debug> = Rc::from(b);

    assert_eq!(format!("{r:?}"), "()");
}

#[test]
fn test_from_vec() {
    let v = vec![1, 2, 3];
    let r: Rc<[u32]> = Rc::from(v);

    assert_eq!(&r[..], [1, 2, 3]);
}

#[test]
fn test_downcast() {
    use std::any::Any;

    let r1: Rc<dyn Any> = Rc::new(i32::MAX);
    let r2: Rc<dyn Any> = Rc::new("abc");

    assert!(r1.clone().downcast::<u32>().is_err());

    let r1i32 = r1.downcast::<i32>();
    assert!(r1i32.is_ok());
    assert_eq!(r1i32.unwrap(), Rc::new(i32::MAX));

    assert!(r2.clone().downcast::<i32>().is_err());

    let r2str = r2.downcast::<&'static str>();
    assert!(r2str.is_ok());
    assert_eq!(r2str.unwrap(), Rc::new("abc"));
}

#[test]
fn test_array_from_slice() {
    let v = vec![1, 2, 3];
    let r: Rc<[u32]> = Rc::from(v);

    let a: Result<Rc<[u32; 3]>, _> = r.clone().try_into();
    assert!(a.is_ok());

    let a: Result<Rc<[u32; 2]>, _> = r.clone().try_into();
    assert!(a.is_err());
}

#[test]
fn test_rc_cyclic_with_zero_refs() {
    struct ZeroRefs {
        inner: Weak<ZeroRefs>,
    }

    let zero_refs = Rc::new_cyclic(|inner| {
        assert_eq!(inner.strong_count(), 0);
        assert!(inner.upgrade().is_none());
        ZeroRefs { inner: Weak::new() }
    });

    assert_eq!(Rc::strong_count(&zero_refs), 1);
    assert_eq!(Rc::weak_count(&zero_refs), 0);
    assert_eq!(zero_refs.inner.strong_count(), 0);
    assert_eq!(zero_refs.inner.weak_count(), 0);
}

#[test]
fn test_rc_cyclic_with_one_ref() {
    struct OneRef {
        inner: Weak<OneRef>,
    }

    let one_ref = Rc::new_cyclic(|inner| {
        assert_eq!(inner.strong_count(), 0);
        assert!(inner.upgrade().is_none());
        OneRef { inner: inner.clone() }
    });

    assert_eq!(Rc::strong_count(&one_ref), 1);
    assert_eq!(Rc::weak_count(&one_ref), 1);

    let one_ref2 = Weak::upgrade(&one_ref.inner).unwrap();
    assert!(Rc::ptr_eq(&one_ref, &one_ref2));

    assert_eq!(one_ref.inner.strong_count(), 2);
    assert_eq!(one_ref.inner.weak_count(), 1);
}

#[test]
fn test_rc_cyclic_with_two_ref() {
    struct TwoRefs {
        inner: Weak<TwoRefs>,
        inner1: Weak<TwoRefs>,
    }

    let two_refs = Rc::new_cyclic(|inner| {
        assert_eq!(inner.strong_count(), 0);
        assert!(inner.upgrade().is_none());
        TwoRefs { inner: inner.clone(), inner1: inner.clone() }
    });

    assert_eq!(Rc::strong_count(&two_refs), 1);
    assert_eq!(Rc::weak_count(&two_refs), 2);

    let two_ref3 = Weak::upgrade(&two_refs.inner).unwrap();
    assert!(Rc::ptr_eq(&two_refs, &two_ref3));

    let two_ref2 = Weak::upgrade(&two_refs.inner1).unwrap();
    assert!(Rc::ptr_eq(&two_refs, &two_ref2));

    assert_eq!(Rc::strong_count(&two_refs), 3);
    assert_eq!(Rc::weak_count(&two_refs), 2);
}

#[test]
fn test_unique_rc_weak() {
    let rc = UniqueRc::new(42);
    let weak = UniqueRc::downgrade(&rc);
    assert!(weak.upgrade().is_none());

    let _rc = UniqueRc::into_rc(rc);
    assert_eq!(*weak.upgrade().unwrap(), 42);
}

#[test]
fn test_unique_rc_drop_weak() {
    let rc = UniqueRc::new(42);
    let weak = UniqueRc::downgrade(&rc);
    mem::drop(weak);

    let rc = UniqueRc::into_rc(rc);
    assert_eq!(*rc, 42);
}

#[test]
fn test_unique_rc_drops_contents() {
    let mut dropped = false;
    struct DropMe<'a>(&'a mut bool);
    impl Drop for DropMe<'_> {
        fn drop(&mut self) {
            *self.0 = true;
        }
    }
    {
        let rc = UniqueRc::new(DropMe(&mut dropped));
        drop(rc);
    }
    assert!(dropped);
}

/// Exercise the non-default allocator usage.
#[test]
fn test_unique_rc_with_alloc_drops_contents() {
    let mut dropped = false;
    struct DropMe<'a>(&'a mut bool);
    impl Drop for DropMe<'_> {
        fn drop(&mut self) {
            *self.0 = true;
        }
    }
    {
        let rc = UniqueRc::new_in(DropMe(&mut dropped), std::alloc::System);
        drop(rc);
    }
    assert!(dropped);
}

#[test]
fn test_unique_rc_weak_clone_holding_ref() {
    let mut v = UniqueRc::new(0u8);
    let w = UniqueRc::downgrade(&v);
    let r = &mut *v;
    let _ = w.clone(); // touch weak count
    *r = 123;
}

#[test]
fn test_unique_rc_unsizing_coercion() {
    let mut rc: UniqueRc<[u8]> = UniqueRc::new([0u8; 3]);
    assert_eq!(rc.len(), 3);
    rc[0] = 123;
    let rc: Rc<[u8]> = UniqueRc::into_rc(rc);
    assert_eq!(*rc, [123, 0, 0]);
}
