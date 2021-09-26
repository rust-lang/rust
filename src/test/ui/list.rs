// run-pass

#![allow(non_camel_case_types)]
// pretty-expanded FIXME #23616

enum list { cons(isize, Box<list>), nil, }

pub fn main() {
    list::cons(10, Box::new(list::cons(11, Box::new(list::cons(12, Box::new(list::nil))))));
}
