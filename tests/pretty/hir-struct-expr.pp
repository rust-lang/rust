#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-struct-expr.pp

struct StructWithSomeFields {
    field_1: i32,
    field_2: i32,
    field_3: i32,
    field_4: i32,
    field_5: i32,
    field_6: i32,
}

fn main() {
    let a =
        StructWithSomeFields {
            field_1: 1,
            field_2: 2,
            field_3: 3,
            field_4: 4,
            field_5: 5,
            field_6: 6 };
    let a = StructWithSomeFields { field_1: 1, field_2: 2, ..a };
}
