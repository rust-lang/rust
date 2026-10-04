//@ run-pass

use std::cell::RefCell;
use std::cmp::Ordering;

struct Evil<'a, 'b> {
    values: RefCell<Vec<&'a str>>,
    to_insert: &'b String,
}

impl<'a, 'b> PartialEq for Evil<'a, 'b> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<'a> PartialOrd for Evil<'a, 'a> {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        self.values.borrow_mut().push(self.to_insert);
        None
    }
}

fn main() {
    let e;
    let values;
    {
        let to_insert = String::from("Hello, world!");
        e = Evil { values: RefCell::new(Vec::new()), to_insert: &to_insert };
        let range = &e..=&e;
        let _ = range == range;
        values = e.values;
    }
    assert_eq!(*values.borrow(), Vec::<&str>::new());
}
