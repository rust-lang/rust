//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
// Make sure validation can handle many overlapping shared borrows for different parts of a data structure
use std::cell::RefCell;

#[allow(unused)]
struct Test {
    a: u32,
    b: u32,
}

fn test1() {
    let t = &mut Test { a: 0, b: 0 };
    {
        let x;
        {
            let y = &t.a;
            x = &t;
            let _y = *y;
        }
        let _x = x.a;
    }
    t.b = 42;
}

fn test2(r: &mut RefCell<i32>) {
    let x = &*r; // releasing write lock, first suspension recorded
    let mut x_ref = x.borrow_mut();
    let x_inner: &mut i32 = &mut *x_ref; // new inner write lock, with same lifetime as outer lock
    let _x_inner_shr = &*x_inner; // releasing inner write lock, recording suspension
    let _y = &*r; // second suspension for the outer write lock
    let _x_inner_shr2 = &*x_inner; // 2nd suspension for inner write lock
}

fn main() {
    test1();
    test2(&mut RefCell::new(0));
}
