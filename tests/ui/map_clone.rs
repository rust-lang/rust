#![warn(clippy::map_clone)]
#![allow(
    clippy::clone_on_copy,
    clippy::iter_cloned_collect,
    clippy::many_single_char_names,
    clippy::redundant_clone,
    clippy::redundant_closure,
    clippy::useless_asref,
    clippy::useless_vec,
    clippy::empty_loop
)]

fn main() {
    let _: Vec<i8> = vec![5_i8; 6].iter().map(|x| *x).collect();
    //~^ map_clone
    let _: Vec<String> = vec![String::new()].iter().map(|x| x.clone()).collect();
    //~^ map_clone
    let _: Vec<u32> = vec![42, 43].iter().map(|&x| x).collect();
    //~^ map_clone
    let _: Option<u64> = Some(Box::new(16)).map(|b| *b);
    let _: Option<u64> = Some(&16).map(|b| *b);
    //~^ map_clone
    let _: Option<u8> = Some(&1).map(|x| x.clone());
    //~^ map_clone

    // Don't lint these
    let v = vec![5_i8; 6];
    let a = 0;
    let b = &a;
    let _ = v.iter().map(|_x| *b);
    let _ = v.iter().map(|_x| a.clone());
    let _ = v.iter().map(|&_x| a);

    // Issue #498
    let _ = std::env::args().map(|v| v.clone());
    //~^ map_clone

    // Issue #4824 item types that aren't references
    {
        use std::rc::Rc;

        let o: Option<Rc<u32>> = Some(Rc::new(0_u32));
        let _: Option<u32> = o.map(|x| *x);
        let v: Vec<Rc<u32>> = vec![Rc::new(0_u32)];
        let _: Vec<u32> = v.into_iter().map(|x| *x).collect();
    }

    // Issue #5524 mutable references
    {
        let mut c = 42;
        let v = vec![&mut c];
        let _: Vec<u32> = v.into_iter().map(|x| *x).collect();
        let mut d = 21;
        let v = vec![&mut d];
        let _: Vec<u32> = v.into_iter().map(|&mut x| x).collect();
    }

    // Issue #6299
    {
        let mut aa = 5;
        let mut bb = 3;
        let items = vec![&mut aa, &mut bb];
        let _: Vec<_> = items.into_iter().map(|x| x.clone()).collect();
    }

    // Issue #6239 deref coercion and clone deref
    {
        use std::cell::RefCell;

        let _ = Some(RefCell::new(String::new()).borrow()).map(|s| s.clone());
    }

    let x = Some(String::new());
    let x = x.as_ref(); // We do this to prevent triggering the `useless_asref` lint.
    let y = x.map(|x| String::clone(x));
    //~^ map_clone

    let y = x.map(Clone::clone);
    //~^ map_clone

    let y = x.map(String::clone);
    //~^ map_clone

    let x: Option<u32> = Some(0);
    let x = x.as_ref(); // We do this to prevent triggering the `useless_asref` lint.
    let y = x.map(|x| u32::clone(x));
    //~^ map_clone

    let y = x.map(|x| Clone::clone(x));
    //~^ map_clone

    // Should not suggest `copied` or `cloned` here since `T` is not a reference.
    let x: Option<u32> = Some(0);
    let y = x.map(|x| u32::clone(&x));
    let y = x.map(|x| Clone::clone(&x));

    // Testing with `Result` now.
    let x: Result<String, ()> = Ok(String::new());
    let x = x.as_ref(); // We do this to prevent triggering the `useless_asref` lint.
    let y = x.map(|x| String::clone(x));
    //~^ map_clone

    let y = x.map(|x| Clone::clone(x));
    //~^ map_clone

    let x: Result<u32, ()> = Ok(0);
    let x = x.as_ref(); // We do this to prevent triggering the `useless_asref` lint.
    let y = x.map(|x| u32::clone(x));
    //~^ map_clone

    let y = x.map(|x| Clone::clone(x));
    //~^ map_clone

    // Should not suggest `copied` or `cloned` here since `T` is not a reference.
    let x: Result<u32, ()> = Ok(0);
    let y = x.map(|x| u32::clone(&x));
    let y = x.map(|x| Clone::clone(&x));

    // We ensure that no warning is emitted here because `useless_asref` is taking over.
    let x = Some(String::new());
    let y = x.as_ref().map(|x| String::clone(x));
    let x: Result<String, ()> = Ok(String::new());
    let y = x.as_ref().map(|x| String::clone(x));

    // Issue #12271
    {
        // Don't lint these
        let x: Option<&u8> = None;
        let y = x.map(|x| String::clone(loop {}));
        let x: Option<&u8> = None;
        let y = x.map(|x| u8::clone(loop {}));
        let x: Vec<&u8> = vec![];
        let y = x.into_iter().map(|x| String::clone(loop {}));
        let x: Vec<&u8> = vec![];
        let y = x.into_iter().map(|x| u8::clone(loop {}));
    }

    // Issue #12528
    {
        // Don't lint these
        use std::rc::{Rc, Weak as RcWeak};
        use std::sync::{Arc, Weak as ArcWeak};
        struct Foo;

        let x = Arc::new(Foo);
        let y = Some(&x);
        let _z = y.map(Arc::clone);

        let x = Rc::new(Foo);
        let y = Some(&x);
        let _z = y.map(Rc::clone);

        let x = Arc::downgrade(&Arc::new(Foo));
        let y = Some(&x);
        let _z = y.map(ArcWeak::clone);

        let x = Rc::downgrade(&Rc::new(Foo));
        let y = Some(&x);
        let _z = y.map(RcWeak::clone);
    }
}
