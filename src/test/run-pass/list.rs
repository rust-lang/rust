#![allow(non_camel_case_types)]
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

enum list { cons(isize, Box<list>), nil, }

pub fn main() { list::cons(10, box list::cons(11, box list::cons(12, box list::nil))); }
