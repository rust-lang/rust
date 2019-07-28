// run-pass
#![allow(non_camel_case_types)]
#![feature(box_syntax)]

enum list<T> { cons(Box<T>, Box<list<T>>), nil, }

pub fn main() {
    let _a: list<isize> =
        list::cons::<isize>(box 10,
        box list::cons::<isize>(box 12,
        box list::cons::<isize>(box 13,
        box list::nil::<isize>)));
}
