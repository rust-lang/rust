//@ run-pass
#![allow(unused_variables)]
#![allow(stable_features)]

use std::cell::RefCell;
use std::rc::Rc;

#[derive(PartialEq, Debug)]
struct Point {
    x: isize,
    y: isize
}

pub fn main() {
    let box_5: Box<_> = Box::new(5_usize);
    let point = Rc::new(Point {x: 2, y: 4});
    assert_eq!(point.x, 2);
    assert_eq!(point.y, 4);

    let i = Rc::new(RefCell::new(2));
    let i_value = *i.borrow();
    *i.borrow_mut() = 5;
    assert_eq!((i_value, *i.borrow()), (2, 5));

    let s = Rc::new("foo".to_string());
    assert_eq!(&**s, "foo");

    let mut_s = Rc::new(RefCell::new(String::from("foo")));
    mut_s.borrow_mut().push_str("bar");
    // HACK assert_eq! would panic here because it stores the LHS and RHS in two locals.
    assert_eq!(&**mut_s.borrow(), "foobar");
    assert_eq!(&**mut_s.borrow_mut(), "foobar");

    let p = Rc::new(RefCell::new(Point {x: 1, y: 2}));
    p.borrow_mut().x = 3;
    p.borrow_mut().y += 3;
    assert_eq!(*p.borrow(), Point {x: 3, y: 5});

    let v = Rc::new(RefCell::new([1, 2, 3]));
    v.borrow_mut()[0] = 3;
    v.borrow_mut()[1] += 3;
    assert_eq!((v.borrow()[0], v.borrow()[1], v.borrow()[2]), (3, 5, 3));
}
