// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unused_variables)]
#![feature(box_syntax)]

enum option<T> { some(Box<T>), none, }

pub fn main() {
    let mut a: option<isize> = option::some::<isize>(box 10);
    a = option::none::<isize>;
}
