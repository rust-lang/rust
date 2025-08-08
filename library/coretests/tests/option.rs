use core::cell::Cell;
use core::mem;
use core::ops::DerefMut;
use core::option::*;

#[test]
fn test_get_ptr() {
    unsafe {
        let x: Box<_> = Box::new(0);
        let addr_x: *const isize = mem::transmute(&*x);
        let opt = Some(x);
        let y = opt.unwrap();
        let addr_y: *const isize = mem::transmute(&*y);
        assert_eq!(addr_x, addr_y);
    }
}

#[test]
fn test_get_str() {
    let x = "test".to_string();
    let addr_x = x.as_ptr();
    let opt = Some(x);
    let y = opt.unwrap();
    let addr_y = y.as_ptr();
    assert_eq!(addr_x, addr_y);
}

#[test]
fn test_get_resource() {
    use core::cell::RefCell;
    use std::rc::Rc;

    struct R {
        i: Rc<RefCell<isize>>,
    }

    impl Drop for R {
        fn drop(&mut self) {
            let ii = &*self.i;
            let i = *ii.borrow();
            *ii.borrow_mut() = i + 1;
        }
    }

    fn r(i: Rc<RefCell<isize>>) -> R {
        R { i }
    }

    let i = Rc::new(RefCell::new(0));
    {
        let x = r(i.clone());
        let opt = Some(x);
        let _y = opt.unwrap();
    }
    assert_eq!(*i.borrow(), 1);
}

#[test]
#[allow(for_loops_over_fallibles)]
fn test_option_dance() {
    let x = Some(());
    let mut y = Some(5);
    let mut y2 = 0;
    for _x in x {
        y2 = y.take().unwrap();
    }
    assert_eq!(y2, 5);
    assert!(y.is_none());
}

#[test]
#[should_panic]
fn test_option_too_much_dance() {
    struct A;
    let mut y = Some(A);
    let _y2 = y.take().unwrap();
    let _y3 = y.take().unwrap();
}

#[test]
fn test_and() {
    let x: Option<isize> = Some(1);
    assert_eq!(x.and(Some(2)), Some(2));
    assert_eq!(x.and(None::<isize>), None);

    let x: Option<isize> = None;
    assert_eq!(x.and(Some(2)), None);
    assert_eq!(x.and(None::<isize>), None);

    const FOO: Option<isize> = Some(1);
    const A: Option<isize> = FOO.and(Some(2));
    const B: Option<isize> = FOO.and(None);
    assert_eq!(A, Some(2));
    assert_eq!(B, None);

    const BAR: Option<isize> = None;
    const C: Option<isize> = BAR.and(Some(2));
    const D: Option<isize> = BAR.and(None);
    assert_eq!(C, None);
    assert_eq!(D, None);
}

#[test]
fn test_and_then() {
    const fn plus_one(x: isize) -> Option<isize> {
        Some(x + 1)
    }

    const fn none(_: isize) -> Option<isize> {
        None
    }

    let x: Option<isize> = Some(1);
    assert_eq!(x.and_then(plus_one), Some(2));
    assert_eq!(x.and_then(none), None);

    let x: Option<isize> = None;
    assert_eq!(x.and_then(plus_one), None);
    assert_eq!(x.and_then(none), None);

    /* FIXME(#110395)
    const FOO: Option<isize> = Some(1);
    const A: Option<isize> = FOO.and_then(plus_one);
    const B: Option<isize> = FOO.and_then(none);
    assert_eq!(A, Some(2));
    assert_eq!(B, None);

    const BAR: Option<isize> = None;
    const C: Option<isize> = BAR.and_then(plus_one);
    const D: Option<isize> = BAR.and_then(none);
    assert_eq!(C, None);
    assert_eq!(D, None);
    */
}

#[test]
fn test_or() {
    let x: Option<isize> = Some(1);
    assert_eq!(x.or(Some(2)), Some(1));
    assert_eq!(x.or(None), Some(1));

    let x: Option<isize> = None;
    assert_eq!(x.or(Some(2)), Some(2));
    assert_eq!(x.or(None), None);

    /* FIXME(#110395)
    const FOO: Option<isize> = Some(1);
    const A: Option<isize> = FOO.or(Some(2));
    const B: Option<isize> = FOO.or(None);
    assert_eq!(A, Some(1));
    assert_eq!(B, Some(1));

    const BAR: Option<isize> = None;
    const C: Option<isize> = BAR.or(Some(2));
    const D: Option<isize> = BAR.or(None);
    assert_eq!(C, Some(2));
    assert_eq!(D, None);
    */
}

#[test]
fn test_or_else() {
    const fn two() -> Option<isize> {
        Some(2)
    }

    const fn none() -> Option<isize> {
        None
    }

    let x: Option<isize> = Some(1);
    assert_eq!(x.or_else(two), Some(1));
    assert_eq!(x.or_else(none), Some(1));

    let x: Option<isize> = None;
    assert_eq!(x.or_else(two), Some(2));
    assert_eq!(x.or_else(none), None);

    /* FIXME(#110395)
        const FOO: Option<isize> = Some(1);
        const A: Option<isize> = FOO.or_else(two);
        const B: Option<isize> = FOO.or_else(none);
        assert_eq!(A, Some(1));
        assert_eq!(B, Some(1));

        const BAR: Option<isize> = None;
        const C: Option<isize> = BAR.or_else(two);
        const D: Option<isize> = BAR.or_else(none);
        assert_eq!(C, Some(2));
        assert_eq!(D, None);
    */
}

#[test]
fn test_unwrap() {
    assert_eq!(Some(1).unwrap(), 1);
    let s = Some("hello".to_string()).unwrap();
    assert_eq!(s, "hello");
}

#[test]
#[should_panic]
fn test_unwrap_panic1() {
    let x: Option<isize> = None;
    x.unwrap();
}

#[test]
#[should_panic]
fn test_unwrap_panic2() {
    let x: Option<String> = None;
    x.unwrap();
}

#[test]
fn test_unwrap_or() {
    let x: Option<isize> = Some(1);
    assert_eq!(x.unwrap_or(2), 1);

    let x: Option<isize> = None;
    assert_eq!(x.unwrap_or(2), 2);

    /* FIXME(#110395)
    const A: isize = Some(1).unwrap_or(2);
    const B: isize = None.unwrap_or(2);
    assert_eq!(A, 1);
    assert_eq!(B, 2);
    */
}

#[test]
fn test_unwrap_or_else() {
    const fn two() -> isize {
        2
    }

    let x: Option<isize> = Some(1);
    assert_eq!(x.unwrap_or_else(two), 1);

    let x: Option<isize> = None;
    assert_eq!(x.unwrap_or_else(two), 2);

    /* FIXME(#110395)
    const A: isize = Some(1).unwrap_or_else(two);
    const B: isize = None.unwrap_or_else(two);
    assert_eq!(A, 1);
    assert_eq!(B, 2);
    */
}

#[test]
fn test_unwrap_unchecked() {
    assert_eq!(unsafe { Some(1).unwrap_unchecked() }, 1);
    let s = unsafe { Some("hello".to_string()).unwrap_unchecked() };
    assert_eq!(s, "hello");
}

#[test]
fn test_iter() {
    let val = 5;

    let x = Some(val);
    let mut it = x.iter();

    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next(), Some(&val));
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());

    let mut it = (&x).into_iter();
    assert_eq!(it.next(), Some(&val));
}

#[test]
fn test_mut_iter() {
    let mut val = 5;
    let new_val = 11;

    let mut x = Some(val);
    {
        let mut it = x.iter_mut();

        assert_eq!(it.size_hint(), (1, Some(1)));

        match it.next() {
            Some(interior) => {
                assert_eq!(*interior, val);
                *interior = new_val;
            }
            None => assert!(false),
        }

        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }
    assert_eq!(x, Some(new_val));

    let mut y = Some(val);
    let mut it = (&mut y).into_iter();
    assert_eq!(it.next(), Some(&mut val));
}

#[test]
fn test_ord() {
    let small = Some(1.0f64);
    let big = Some(5.0f64);
    let nan = Some(0.0f64 / 0.0);
    assert!(!(nan < big));
    assert!(!(nan > big));
    assert!(small < big);
    assert!(None < big);
    assert!(big > None);
}

#[test]
fn test_collect() {
    let v: Option<Vec<isize>> = (0..0).map(|_| Some(0)).collect();
    assert!(v == Some(vec![]));

    let v: Option<Vec<isize>> = (0..3).map(|x| Some(x)).collect();
    assert!(v == Some(vec![0, 1, 2]));

    let v: Option<Vec<isize>> = (0..3).map(|x| if x > 1 { None } else { Some(x) }).collect();
    assert!(v == None);

    // test that it does not take more elements than it needs
    let mut functions: [Box<dyn Fn() -> Option<()>>; 3] =
        [Box::new(|| Some(())), Box::new(|| None), Box::new(|| panic!())];

    let v: Option<Vec<()>> = functions.iter_mut().map(|f| (*f)()).collect();

    assert!(v == None);
}

#[test]
fn test_copied() {
    let val = 1;
    let val_ref = &val;
    let opt_none: Option<&'static u32> = None;
    let opt_ref = Some(&val);
    let opt_ref_ref = Some(&val_ref);

    // None works
    assert_eq!(opt_none.clone(), None);
    assert_eq!(opt_none.copied(), None);

    // Immutable ref works
    assert_eq!(opt_ref.clone(), Some(&val));
    assert_eq!(opt_ref.copied(), Some(1));

    // Double Immutable ref works
    assert_eq!(opt_ref_ref.clone(), Some(&val_ref));
    assert_eq!(opt_ref_ref.clone().copied(), Some(&val));
    assert_eq!(opt_ref_ref.copied().copied(), Some(1));
}

#[test]
fn test_cloned() {
    let val = 1;
    let val_ref = &val;
    let opt_none: Option<&'static u32> = None;
    let opt_ref = Some(&val);
    let opt_ref_ref = Some(&val_ref);

    // None works
    assert_eq!(opt_none.clone(), None);
    assert_eq!(opt_none.cloned(), None);

    // Immutable ref works
    assert_eq!(opt_ref.clone(), Some(&val));
    assert_eq!(opt_ref.cloned(), Some(1));

    // Double Immutable ref works
    assert_eq!(opt_ref_ref.clone(), Some(&val_ref));
    assert_eq!(opt_ref_ref.clone().cloned(), Some(&val));
    assert_eq!(opt_ref_ref.cloned().cloned(), Some(1));
}

#[test]
fn test_try() {
    fn try_option_some() -> Option<u8> {
        let val = Some(1)?;
        Some(val)
    }
    assert_eq!(try_option_some(), Some(1));

    fn try_option_none() -> Option<u8> {
        let val = None?;
        Some(val)
    }
    assert_eq!(try_option_none(), None);
}

#[test]
fn test_option_as_deref() {
    // Some: &Option<T: Deref>::Some(T) -> Option<&T::Deref::Target>::Some(&*T)
    let ref_option = &Some(&42);
    assert_eq!(ref_option.as_deref(), Some(&42));

    let ref_option = &Some(String::from("a result"));
    assert_eq!(ref_option.as_deref(), Some("a result"));

    let ref_option = &Some(vec![1, 2, 3, 4, 5]);
    assert_eq!(ref_option.as_deref(), Some([1, 2, 3, 4, 5].as_slice()));

    // None: &Option<T: Deref>>::None -> None
    let ref_option: &Option<&i32> = &None;
    assert_eq!(ref_option.as_deref(), None);
}

#[test]
fn test_option_as_deref_mut() {
    // Some: &mut Option<T: Deref>::Some(T) -> Option<&mut T::Deref::Target>::Some(&mut *T)
    let mut val = 42;
    let ref_option = &mut Some(&mut val);
    assert_eq!(ref_option.as_deref_mut(), Some(&mut 42));

    let ref_option = &mut Some(String::from("a result"));
    assert_eq!(ref_option.as_deref_mut(), Some(String::from("a result").deref_mut()));

    let ref_option = &mut Some(vec![1, 2, 3, 4, 5]);
    assert_eq!(ref_option.as_deref_mut(), Some([1, 2, 3, 4, 5].as_mut_slice()));

    // None: &mut Option<T: Deref>>::None -> None
    let ref_option: &mut Option<&mut i32> = &mut None;
    assert_eq!(ref_option.as_deref_mut(), None);
}

#[test]
fn test_replace() {
    let mut x = Some(2);
    let old = x.replace(5);

    assert_eq!(x, Some(5));
    assert_eq!(old, Some(2));

    let mut x = None;
    let old = x.replace(3);

    assert_eq!(x, Some(3));
    assert_eq!(old, None);
}

#[test]
fn option_const() {
    // test that the methods of `Option` are usable in a const context

    const OPTION: Option<usize> = Some(32);
    assert_eq!(OPTION, Some(32));

    // FIXME(#110395)
    // const OPTION_FROM: Option<usize> = Option::from(32);
    // assert_eq!(OPTION_FROM, Some(32));

    const REF: Option<&usize> = OPTION.as_ref();
    assert_eq!(REF, Some(&32));

    // const REF_FROM: Option<&usize> = Option::from(&OPTION);
    // assert_eq!(REF_FROM, Some(&32));

    const IS_SOME: bool = OPTION.is_some();
    assert!(IS_SOME);

    const IS_NONE: bool = OPTION.is_none();
    assert!(!IS_NONE);

    const COPIED: Option<usize> = OPTION.as_ref().copied();
    assert_eq!(COPIED, OPTION);
}

#[test]
const fn option_const_mut() {
    // test that the methods of `Option` that take mutable references are usable in a const context

    let mut option: Option<usize> = Some(32);

    let _take = option.take();
    let _replace = option.replace(42);

    {
        let as_mut = option.as_mut();
        match as_mut {
            Some(v) => *v = 32,
            None => unreachable!(),
        }
    }
    /* FIXME(const-hack)
        {
            let as_mut: Option<&mut usize> = Option::from(&mut option);
            match as_mut {
                Some(v) => *v = 42,
                None => unreachable!(),
            }
        }
    */
}

#[test]
fn test_unwrap_drop() {
    struct Dtor<'a> {
        x: &'a Cell<isize>,
    }

    impl<'a> std::ops::Drop for Dtor<'a> {
        fn drop(&mut self) {
            self.x.set(self.x.get() - 1);
        }
    }

    fn unwrap<T>(o: Option<T>) -> T {
        match o {
            Some(v) => v,
            None => panic!(),
        }
    }

    let x = &Cell::new(1);

    {
        let b = Some(Dtor { x });
        let _c = unwrap(b);
    }

    assert_eq!(x.get(), 0);
}

#[test]
fn option_ext() {
    let thing = "{{ f }}";
    let f = thing.find("{{");

    if f.is_none() {
        println!("None!");
    }
}

#[test]
fn zip_options() {
    let x = Some(10);
    let y = Some("foo");
    let z: Option<usize> = None;

    assert_eq!(x.zip(y), Some((10, "foo")));
    assert_eq!(x.zip(z), None);
    assert_eq!(z.zip(x), None);
}

#[test]
fn unzip_options() {
    let x = Some((10, "foo"));
    let y = None::<(bool, i32)>;

    assert_eq!(x.unzip(), (Some(10), Some("foo")));
    assert_eq!(y.unzip(), (None, None));
}

#[test]
fn zip_unzip_roundtrip() {
    let x = Some(10);
    let y = Some("foo");

    let z = x.zip(y);
    assert_eq!(z, Some((10, "foo")));

    let a = z.unzip();
    assert_eq!(a, (x, y));
}

#[test]
fn as_slice() {
    assert_eq!(Some(42).as_slice(), &[42]);
    assert_eq!(Some(43).as_mut_slice(), &[43]);
    assert_eq!(None::<i32>.as_slice(), &[]);
    assert_eq!(None::<i32>.as_mut_slice(), &[]);

    const A: &[u32] = Some(44).as_slice();
    const B: &[u32] = None.as_slice();
    const _: () = {
        let [45] = Some(45).as_mut_slice() else { panic!() };
        let []: &[u32] = None.as_mut_slice() else { panic!() };
    };
    assert_eq!(A, &[44]);
    assert_eq!(B, &[]);
}
