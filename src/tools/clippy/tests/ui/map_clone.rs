// run-rustfix
#![warn(clippy::map_clone)]
#![allow(
    clippy::clone_on_copy,
    clippy::iter_cloned_collect,
    clippy::many_single_char_names,
    clippy::redundant_clone
)]

fn main() {
    let _: Vec<i8> = vec![5_i8; 6].iter().map(|x| *x).collect();
    let _: Vec<String> = vec![String::new()].iter().map(|x| x.clone()).collect();
    let _: Vec<u32> = vec![42, 43].iter().map(|&x| x).collect();
    let _: Option<u64> = Some(Box::new(16)).map(|b| *b);
    let _: Option<u64> = Some(&16).map(|b| *b);
    let _: Option<u8> = Some(&1).map(|x| x.clone());

    // Don't lint these
    let v = vec![5_i8; 6];
    let a = 0;
    let b = &a;
    let _ = v.iter().map(|_x| *b);
    let _ = v.iter().map(|_x| a.clone());
    let _ = v.iter().map(|&_x| a);

    // Issue #498
    let _ = std::env::args().map(|v| v.clone());

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
}
