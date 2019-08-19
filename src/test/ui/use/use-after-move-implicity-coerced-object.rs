#![feature(box_syntax)]

use std::fmt;

struct Number {
    n: i64
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.n)
    }
}

struct List {
    list: Vec<Box<dyn ToString + 'static>> }

impl List {
    fn push(&mut self, n: Box<dyn ToString + 'static>) {
        self.list.push(n);
    }
}

fn main() {
    let n: Box<_> = box Number { n: 42 };
    let mut l: Box<_> = box List { list: Vec::new() };
    l.push(n);
    let x = n.to_string();
    //~^ ERROR: borrow of moved value: `n`
}
