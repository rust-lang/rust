// run-pass
#![allow(unused_assignments)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

#![allow(unused_variables)]

enum option<T> { some(#[allow(unused_tuple_struct_fields)] Box<T>), none, }

pub fn main() {
    let mut a: option<isize> = option::some::<isize>(Box::new(10));
    a = option::none::<isize>;
}
