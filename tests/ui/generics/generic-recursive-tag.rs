//@ run-pass
#![allow(non_camel_case_types)]

enum list<T> { #[allow(dead_code)] cons(Box<T>, Box<list<T>>), nil, }

pub fn main() {
    let _a: list<isize> =
        list::cons::<isize>(Box::new(10),
        Box::new(list::cons::<isize>(Box::new(12),
        Box::new(list::cons::<isize>(Box::new(13),
        Box::new(list::nil::<isize>))))));
}
