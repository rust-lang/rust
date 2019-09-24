// run-pass
#![allow(unused_assignments)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

#![allow(unused_variables)]
#![feature(box_syntax)]

enum option<T> { some(Box<T>), none, }

pub fn main() {
    let mut a: option<isize> = option::some::<isize>(box 10);
    a = option::none::<isize>;
}
