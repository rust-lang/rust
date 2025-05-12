//@ run-pass

#![feature(const_type_name)]
#![allow(dead_code)]

const fn type_name_wrapper<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}

struct Struct<TA, TB, TC> {
    a: TA,
    b: TB,
    c: TC,
}

type StructInstantiation = Struct<i8, f64, bool>;

const CONST_STRUCT: StructInstantiation = StructInstantiation { a: 12, b: 13.7, c: false };

const CONST_STRUCT_NAME: &'static str = type_name_wrapper(&CONST_STRUCT);

fn main() {
    let non_const_struct = StructInstantiation { a: 87, b: 65.99, c: true };

    let non_const_struct_name = type_name_wrapper(&non_const_struct);

    assert_eq!(CONST_STRUCT_NAME, non_const_struct_name);
}
