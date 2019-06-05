mod owning_ref {
    use super::super::OwningRef;
    use super::super::{RcRef, BoxRef, Erased, ErasedBoxRef};
    use std::cmp::{PartialEq, Ord, PartialOrd, Ordering};
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[derive(Debug, PartialEq)]
    struct Example(u32, String, [u8; 3]);
    fn example() -> Example {
        Example(42, "hello world".to_string(), [1, 2, 3])
    }

    #[test]
    fn new_deref() {
        let or: OwningRef<Box<()>, ()> = OwningRef::new(Box::new(()));
        assert_eq!(&*or, &());
    }

    #[test]
    fn into() {
        let or: OwningRef<Box<()>, ()> = Box::new(()).into();
        assert_eq!(&*or, &());
    }

    #[test]
    fn map_offset_ref() {
        let or: BoxRef<Example> = Box::new(example()).into();
        let or: BoxRef<_, u32> = or.map(|x| &x.0);
        assert_eq!(&*or, &42);

        let or: BoxRef<Example> = Box::new(example()).into();
        let or: BoxRef<_, u8> = or.map(|x| &x.2[1]);
        assert_eq!(&*or, &2);
    }

    #[test]
    fn map_heap_ref() {
        let or: BoxRef<Example> = Box::new(example()).into();
        let or: BoxRef<_, str> = or.map(|x| &x.1[..5]);
        assert_eq!(&*or, "hello");
    }

    #[test]
    fn map_static_ref() {
        let or: BoxRef<()> = Box::new(()).into();
        let or: BoxRef<_, str> = or.map(|_| "hello");
        assert_eq!(&*or, "hello");
    }

    #[test]
    fn map_chained() {
        let or: BoxRef<String> = Box::new(example().1).into();
        let or: BoxRef<_, str> = or.map(|x| &x[1..5]);
        let or: BoxRef<_, str> = or.map(|x| &x[..2]);
        assert_eq!(&*or, "el");
    }

    #[test]
    fn map_chained_inference() {
        let or = BoxRef::new(Box::new(example().1))
            .map(|x| &x[..5])
            .map(|x| &x[1..3]);
        assert_eq!(&*or, "el");
    }

    #[test]
    fn owner() {
        let or: BoxRef<String> = Box::new(example().1).into();
        let or = or.map(|x| &x[..5]);
        assert_eq!(&*or, "hello");
        assert_eq!(&**or.owner(), "hello world");
    }

    #[test]
    fn into_inner() {
        let or: BoxRef<String> = Box::new(example().1).into();
        let or = or.map(|x| &x[..5]);
        assert_eq!(&*or, "hello");
        let s = *or.into_inner();
        assert_eq!(&s, "hello world");
    }

    #[test]
    fn fmt_debug() {
        let or: BoxRef<String> = Box::new(example().1).into();
        let or = or.map(|x| &x[..5]);
        let s = format!("{:?}", or);
        assert_eq!(&s, "OwningRef { owner: \"hello world\", reference: \"hello\" }");
    }

    #[test]
    fn erased_owner() {
        let o1: BoxRef<Example, str> = BoxRef::new(Box::new(example()))
            .map(|x| &x.1[..]);

        let o2: BoxRef<String, str> = BoxRef::new(Box::new(example().1))
            .map(|x| &x[..]);

        let os: Vec<ErasedBoxRef<str>> = vec![o1.erase_owner(), o2.erase_owner()];
        assert!(os.iter().all(|e| &e[..] == "hello world"));
    }

    #[test]
    fn raii_locks() {
        use super::super::{RefRef, RefMutRef};
        use std::cell::RefCell;
        use super::super::{MutexGuardRef, RwLockReadGuardRef, RwLockWriteGuardRef};
        use std::sync::{Mutex, RwLock};

        {
            let a = RefCell::new(1);
            let a = {
                let a = RefRef::new(a.borrow());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
        {
            let a = RefCell::new(1);
            let a = {
                let a = RefMutRef::new(a.borrow_mut());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
        {
            let a = Mutex::new(1);
            let a = {
                let a = MutexGuardRef::new(a.lock().unwrap());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
        {
            let a = RwLock::new(1);
            let a = {
                let a = RwLockReadGuardRef::new(a.read().unwrap());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
        {
            let a = RwLock::new(1);
            let a = {
                let a = RwLockWriteGuardRef::new(a.write().unwrap());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
    }

    #[test]
    fn eq() {
        let or1: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
        let or2: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
        assert_eq!(or1.eq(&or2), true);
    }

    #[test]
    fn cmp() {
        let or1: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
        let or2: BoxRef<[u8]> = BoxRef::new(vec![4, 5, 6].into_boxed_slice());
        assert_eq!(or1.cmp(&or2), Ordering::Less);
    }

    #[test]
    fn partial_cmp() {
        let or1: BoxRef<[u8]> = BoxRef::new(vec![4, 5, 6].into_boxed_slice());
        let or2: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
        assert_eq!(or1.partial_cmp(&or2), Some(Ordering::Greater));
    }

    #[test]
    fn hash() {
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();

        let or1: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
        let or2: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());

        or1.hash(&mut h1);
        or2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn borrow() {
        let mut hash = HashMap::new();
        let     key  = RcRef::<String>::new(Rc::new("foo-bar".to_string())).map(|s| &s[..]);

        hash.insert(key.clone().map(|s| &s[..3]), 42);
        hash.insert(key.clone().map(|s| &s[4..]), 23);

        assert_eq!(hash.get("foo"), Some(&42));
        assert_eq!(hash.get("bar"), Some(&23));
    }

    #[test]
    fn total_erase() {
        let a: OwningRef<Vec<u8>, [u8]>
            = OwningRef::new(vec![]).map(|x| &x[..]);
        let b: OwningRef<Box<[u8]>, [u8]>
            = OwningRef::new(vec![].into_boxed_slice()).map(|x| &x[..]);

        let c: OwningRef<Rc<Vec<u8>>, [u8]> = unsafe {a.map_owner(Rc::new)};
        let d: OwningRef<Rc<Box<[u8]>>, [u8]> = unsafe {b.map_owner(Rc::new)};

        let e: OwningRef<Rc<dyn Erased>, [u8]> = c.erase_owner();
        let f: OwningRef<Rc<dyn Erased>, [u8]> = d.erase_owner();

        let _g = e.clone();
        let _h = f.clone();
    }

    #[test]
    fn total_erase_box() {
        let a: OwningRef<Vec<u8>, [u8]>
            = OwningRef::new(vec![]).map(|x| &x[..]);
        let b: OwningRef<Box<[u8]>, [u8]>
            = OwningRef::new(vec![].into_boxed_slice()).map(|x| &x[..]);

        let c: OwningRef<Box<Vec<u8>>, [u8]> = a.map_owner_box();
        let d: OwningRef<Box<Box<[u8]>>, [u8]> = b.map_owner_box();

        let _e: OwningRef<Box<dyn Erased>, [u8]> = c.erase_owner();
        let _f: OwningRef<Box<dyn Erased>, [u8]> = d.erase_owner();
    }

    #[test]
    fn try_map1() {
        use std::any::Any;

        let x = Box::new(123_i32);
        let y: Box<dyn Any> = x;

        assert!(OwningRef::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_ok());
    }

    #[test]
    fn try_map2() {
        use std::any::Any;

        let x = Box::new(123_i32);
        let y: Box<dyn Any> = x;

        assert!(!OwningRef::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_err());
    }
}

mod owning_handle {
    use super::super::OwningHandle;
    use super::super::RcRef;
    use std::rc::Rc;
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::RwLock;

    #[test]
    fn owning_handle() {
        use std::cell::RefCell;
        let cell = Rc::new(RefCell::new(2));
        let cell_ref = RcRef::new(cell);
        let mut handle = OwningHandle::new_with_fn(cell_ref, |x| unsafe { x.as_ref() }.unwrap().borrow_mut());
        assert_eq!(*handle, 2);
        *handle = 3;
        assert_eq!(*handle, 3);
    }

    #[test]
    fn try_owning_handle_ok() {
        use std::cell::RefCell;
        let cell = Rc::new(RefCell::new(2));
        let cell_ref = RcRef::new(cell);
        let mut handle = OwningHandle::try_new::<_, ()>(cell_ref, |x| {
            Ok(unsafe {
                x.as_ref()
            }.unwrap().borrow_mut())
        }).unwrap();
        assert_eq!(*handle, 2);
        *handle = 3;
        assert_eq!(*handle, 3);
    }

    #[test]
    fn try_owning_handle_err() {
        use std::cell::RefCell;
        let cell = Rc::new(RefCell::new(2));
        let cell_ref = RcRef::new(cell);
        let handle = OwningHandle::try_new::<_, ()>(cell_ref, |x| {
            if false {
                return Ok(unsafe {
                    x.as_ref()
                }.unwrap().borrow_mut())
            }
            Err(())
        });
        assert!(handle.is_err());
    }

    #[test]
    fn nested() {
        use std::cell::RefCell;
        use std::sync::{Arc, RwLock};

        let result = {
            let complex = Rc::new(RefCell::new(Arc::new(RwLock::new("someString"))));
            let curr = RcRef::new(complex);
            let curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().borrow_mut());
            let mut curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().try_write().unwrap());
            assert_eq!(*curr, "someString");
            *curr = "someOtherString";
            curr
        };
        assert_eq!(*result, "someOtherString");
    }

    #[test]
    fn owning_handle_safe() {
        use std::cell::RefCell;
        let cell = Rc::new(RefCell::new(2));
        let cell_ref = RcRef::new(cell);
        let handle = OwningHandle::new(cell_ref);
        assert_eq!(*handle, 2);
    }

    #[test]
    fn owning_handle_mut_safe() {
        use std::cell::RefCell;
        let cell = Rc::new(RefCell::new(2));
        let cell_ref = RcRef::new(cell);
        let mut handle = OwningHandle::new_mut(cell_ref);
        assert_eq!(*handle, 2);
        *handle = 3;
        assert_eq!(*handle, 3);
    }

    #[test]
    fn owning_handle_safe_2() {
        let result = {
            let complex = Rc::new(RefCell::new(Arc::new(RwLock::new("someString"))));
            let curr = RcRef::new(complex);
            let curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().borrow_mut());
            let mut curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().try_write().unwrap());
            assert_eq!(*curr, "someString");
            *curr = "someOtherString";
            curr
        };
        assert_eq!(*result, "someOtherString");
    }
}

mod owning_ref_mut {
    use super::super::{OwningRefMut, BoxRefMut, Erased, ErasedBoxRefMut};
    use super::super::BoxRef;
    use std::cmp::{PartialEq, Ord, PartialOrd, Ordering};
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;

    #[derive(Debug, PartialEq)]
    struct Example(u32, String, [u8; 3]);
    fn example() -> Example {
        Example(42, "hello world".to_string(), [1, 2, 3])
    }

    #[test]
    fn new_deref() {
        let or: OwningRefMut<Box<()>, ()> = OwningRefMut::new(Box::new(()));
        assert_eq!(&*or, &());
    }

    #[test]
    fn new_deref_mut() {
        let mut or: OwningRefMut<Box<()>, ()> = OwningRefMut::new(Box::new(()));
        assert_eq!(&mut *or, &mut ());
    }

    #[test]
    fn mutate() {
        let mut or: OwningRefMut<Box<usize>, usize> = OwningRefMut::new(Box::new(0));
        assert_eq!(&*or, &0);
        *or = 1;
        assert_eq!(&*or, &1);
    }

    #[test]
    fn into() {
        let or: OwningRefMut<Box<()>, ()> = Box::new(()).into();
        assert_eq!(&*or, &());
    }

    #[test]
    fn map_offset_ref() {
        let or: BoxRefMut<Example> = Box::new(example()).into();
        let or: BoxRef<_, u32> = or.map(|x| &mut x.0);
        assert_eq!(&*or, &42);

        let or: BoxRefMut<Example> = Box::new(example()).into();
        let or: BoxRef<_, u8> = or.map(|x| &mut x.2[1]);
        assert_eq!(&*or, &2);
    }

    #[test]
    fn map_heap_ref() {
        let or: BoxRefMut<Example> = Box::new(example()).into();
        let or: BoxRef<_, str> = or.map(|x| &mut x.1[..5]);
        assert_eq!(&*or, "hello");
    }

    #[test]
    fn map_static_ref() {
        let or: BoxRefMut<()> = Box::new(()).into();
        let or: BoxRef<_, str> = or.map(|_| "hello");
        assert_eq!(&*or, "hello");
    }

    #[test]
    fn map_mut_offset_ref() {
        let or: BoxRefMut<Example> = Box::new(example()).into();
        let or: BoxRefMut<_, u32> = or.map_mut(|x| &mut x.0);
        assert_eq!(&*or, &42);

        let or: BoxRefMut<Example> = Box::new(example()).into();
        let or: BoxRefMut<_, u8> = or.map_mut(|x| &mut x.2[1]);
        assert_eq!(&*or, &2);
    }

    #[test]
    fn map_mut_heap_ref() {
        let or: BoxRefMut<Example> = Box::new(example()).into();
        let or: BoxRefMut<_, str> = or.map_mut(|x| &mut x.1[..5]);
        assert_eq!(&*or, "hello");
    }

    #[test]
    fn map_mut_static_ref() {
        static mut MUT_S: [u8; 5] = *b"hello";

        let mut_s: &'static mut [u8] = unsafe { &mut MUT_S };

        let or: BoxRefMut<()> = Box::new(()).into();
        let or: BoxRefMut<_, [u8]> = or.map_mut(move |_| mut_s);
        assert_eq!(&*or, b"hello");
    }

    #[test]
    fn map_mut_chained() {
        let or: BoxRefMut<String> = Box::new(example().1).into();
        let or: BoxRefMut<_, str> = or.map_mut(|x| &mut x[1..5]);
        let or: BoxRefMut<_, str> = or.map_mut(|x| &mut x[..2]);
        assert_eq!(&*or, "el");
    }

    #[test]
    fn map_chained_inference() {
        let or = BoxRefMut::new(Box::new(example().1))
            .map_mut(|x| &mut x[..5])
            .map_mut(|x| &mut x[1..3]);
        assert_eq!(&*or, "el");
    }

    #[test]
    fn try_map_mut() {
        let or: BoxRefMut<String> = Box::new(example().1).into();
        let or: Result<BoxRefMut<_, str>, ()> = or.try_map_mut(|x| Ok(&mut x[1..5]));
        assert_eq!(&*or.unwrap(), "ello");

        let or: BoxRefMut<String> = Box::new(example().1).into();
        let or: Result<BoxRefMut<_, str>, ()> = or.try_map_mut(|_| Err(()));
        assert!(or.is_err());
    }

    #[test]
    fn owner() {
        let or: BoxRefMut<String> = Box::new(example().1).into();
        let or = or.map_mut(|x| &mut x[..5]);
        assert_eq!(&*or, "hello");
        assert_eq!(&**or.owner(), "hello world");
    }

    #[test]
    fn into_inner() {
        let or: BoxRefMut<String> = Box::new(example().1).into();
        let or = or.map_mut(|x| &mut x[..5]);
        assert_eq!(&*or, "hello");
        let s = *or.into_inner();
        assert_eq!(&s, "hello world");
    }

    #[test]
    fn fmt_debug() {
        let or: BoxRefMut<String> = Box::new(example().1).into();
        let or = or.map_mut(|x| &mut x[..5]);
        let s = format!("{:?}", or);
        assert_eq!(&s,
                   "OwningRefMut { owner: \"hello world\", reference: \"hello\" }");
    }

    #[test]
    fn erased_owner() {
        let o1: BoxRefMut<Example, str> = BoxRefMut::new(Box::new(example()))
            .map_mut(|x| &mut x.1[..]);

        let o2: BoxRefMut<String, str> = BoxRefMut::new(Box::new(example().1))
            .map_mut(|x| &mut x[..]);

        let os: Vec<ErasedBoxRefMut<str>> = vec![o1.erase_owner(), o2.erase_owner()];
        assert!(os.iter().all(|e| &e[..] == "hello world"));
    }

    #[test]
    fn raii_locks() {
        use super::super::RefMutRefMut;
        use std::cell::RefCell;
        use super::super::{MutexGuardRefMut, RwLockWriteGuardRefMut};
        use std::sync::{Mutex, RwLock};

        {
            let a = RefCell::new(1);
            let a = {
                let a = RefMutRefMut::new(a.borrow_mut());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
        {
            let a = Mutex::new(1);
            let a = {
                let a = MutexGuardRefMut::new(a.lock().unwrap());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
        {
            let a = RwLock::new(1);
            let a = {
                let a = RwLockWriteGuardRefMut::new(a.write().unwrap());
                assert_eq!(*a, 1);
                a
            };
            assert_eq!(*a, 1);
            drop(a);
        }
    }

    #[test]
    fn eq() {
        let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
        let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
        assert_eq!(or1.eq(&or2), true);
    }

    #[test]
    fn cmp() {
        let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
        let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![4, 5, 6].into_boxed_slice());
        assert_eq!(or1.cmp(&or2), Ordering::Less);
    }

    #[test]
    fn partial_cmp() {
        let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![4, 5, 6].into_boxed_slice());
        let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
        assert_eq!(or1.partial_cmp(&or2), Some(Ordering::Greater));
    }

    #[test]
    fn hash() {
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();

        let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
        let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());

        or1.hash(&mut h1);
        or2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn borrow() {
        let mut hash = HashMap::new();
        let     key1 = BoxRefMut::<String>::new(Box::new("foo".to_string())).map(|s| &s[..]);
        let     key2 = BoxRefMut::<String>::new(Box::new("bar".to_string())).map(|s| &s[..]);

        hash.insert(key1, 42);
        hash.insert(key2, 23);

        assert_eq!(hash.get("foo"), Some(&42));
        assert_eq!(hash.get("bar"), Some(&23));
    }

    #[test]
    fn total_erase() {
        let a: OwningRefMut<Vec<u8>, [u8]>
            = OwningRefMut::new(vec![]).map_mut(|x| &mut x[..]);
        let b: OwningRefMut<Box<[u8]>, [u8]>
            = OwningRefMut::new(vec![].into_boxed_slice()).map_mut(|x| &mut x[..]);

        let c: OwningRefMut<Box<Vec<u8>>, [u8]> = unsafe {a.map_owner(Box::new)};
        let d: OwningRefMut<Box<Box<[u8]>>, [u8]> = unsafe {b.map_owner(Box::new)};

        let _e: OwningRefMut<Box<dyn Erased>, [u8]> = c.erase_owner();
        let _f: OwningRefMut<Box<dyn Erased>, [u8]> = d.erase_owner();
    }

    #[test]
    fn total_erase_box() {
        let a: OwningRefMut<Vec<u8>, [u8]>
            = OwningRefMut::new(vec![]).map_mut(|x| &mut x[..]);
        let b: OwningRefMut<Box<[u8]>, [u8]>
            = OwningRefMut::new(vec![].into_boxed_slice()).map_mut(|x| &mut x[..]);

        let c: OwningRefMut<Box<Vec<u8>>, [u8]> = a.map_owner_box();
        let d: OwningRefMut<Box<Box<[u8]>>, [u8]> = b.map_owner_box();

        let _e: OwningRefMut<Box<dyn Erased>, [u8]> = c.erase_owner();
        let _f: OwningRefMut<Box<dyn Erased>, [u8]> = d.erase_owner();
    }

    #[test]
    fn try_map1() {
        use std::any::Any;

        let x = Box::new(123_i32);
        let y: Box<dyn Any> = x;

        assert!(OwningRefMut::new(y).try_map_mut(|x| x.downcast_mut::<i32>().ok_or(())).is_ok());
    }

    #[test]
    fn try_map2() {
        use std::any::Any;

        let x = Box::new(123_i32);
        let y: Box<dyn Any> = x;

        assert!(!OwningRefMut::new(y).try_map_mut(|x| x.downcast_mut::<i32>().ok_or(())).is_err());
    }

    #[test]
    fn try_map3() {
        use std::any::Any;

        let x = Box::new(123_i32);
        let y: Box<dyn Any> = x;

        assert!(OwningRefMut::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_ok());
    }

    #[test]
    fn try_map4() {
        use std::any::Any;

        let x = Box::new(123_i32);
        let y: Box<dyn Any> = x;

        assert!(!OwningRefMut::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_err());
    }

    #[test]
    fn into_owning_ref() {
        use super::super::BoxRef;

        let or: BoxRefMut<()> = Box::new(()).into();
        let or: BoxRef<()> = or.into();
        assert_eq!(&*or, &());
    }

    struct Foo {
        u: u32,
    }
    struct Bar {
        f: Foo,
    }

    #[test]
    fn ref_mut() {
        use std::cell::RefCell;

        let a = RefCell::new(Bar { f: Foo { u: 42 } });
        let mut b = OwningRefMut::new(a.borrow_mut());
        assert_eq!(b.f.u, 42);
        b.f.u = 43;
        let mut c = b.map_mut(|x| &mut x.f);
        assert_eq!(c.u, 43);
        c.u = 44;
        let mut d = c.map_mut(|x| &mut x.u);
        assert_eq!(*d, 44);
        *d = 45;
        assert_eq!(*d, 45);
    }
}
