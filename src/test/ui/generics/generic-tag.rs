// run-pass
#![allow(unused_assignments, non_camel_case_types, type_param_on_variant_ctor)]

// pretty-expanded FIXME #23616

#![allow(unused_variables)]
#![feature(box_syntax)]

enum option<T> { some(Box<T>), none, }

pub fn main() {
    let mut a: option<isize> = option::some::<isize>(box 10);
    a = option::none::<isize>;
}
