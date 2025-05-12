// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

#![allow(warnings)]
#![feature(rustc_attrs)]

struct MyStruct {
    field: String
}

fn main() {
}

fn nll_fail() {
    let mut my_struct = MyStruct { field: format!("Hello") };

    let value = &mut my_struct.field;
    loop {
        my_struct.field.push_str("Hello, world!");
        //~^ ERROR [E0499]
        value.len();
        return;
    }
}

fn nll_ok() {
    let mut my_struct = MyStruct { field: format!("Hello") };

    let value = &mut my_struct.field;
    loop {
        my_struct.field.push_str("Hello, world!");
        return;
    }
}
