// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

// compile-flags:-Zborrowck=compare

#![allow(warnings)]
#![feature(rustc_attrs)]

struct MyStruct {
    field: String
}

fn foo1() {
    let mut my_struct = MyStruct { field: format!("Hello") };

    let value = &my_struct.field;
    if value.is_empty() {
        my_struct.field.push_str("Hello, world!");
        //~^ ERROR (Ast) [E0502]
    }
}

fn foo2() {
    let mut my_struct = MyStruct { field: format!("Hello") };

    let value = &my_struct.field;
    if value.is_empty() {
        my_struct.field.push_str("Hello, world!");
        //~^ ERROR (Ast) [E0502]
        //~| ERROR (Mir) [E0502]
    }
    drop(value);
}

fn main() { }
