//@ run-pass
#![allow(unused_assignments)]
#![allow(non_camel_case_types)]


#![allow(unused_variables)]

enum option<T> { some(#[allow(dead_code)] Box<T>), none, }

pub fn main() {
    let mut a: option<isize> = option::some::<isize>(Box::new(10));
    a = option::none::<isize>;
}
