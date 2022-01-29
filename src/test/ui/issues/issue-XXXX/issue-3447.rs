// run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![feature(box_syntax)]

use std::cell::RefCell;

static S: &'static str = "str";

struct list<T> {
    element: T,
    next: Option<Box<RefCell<list<T>>>>
}

impl<T:'static> list<T> {
    pub fn addEnd(&mut self, element: T) {
        let newList = list {
            element: element,
            next: None
        };

        self.next = Some(box RefCell::new(newList));
    }
}

pub fn main() {
    let ls = list {
        element: S,
        next: None
    };
    println!("{}", ls.element);
}
