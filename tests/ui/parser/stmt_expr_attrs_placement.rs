#![feature(stmt_expr_attributes)]

// Test that various placements of the inner attribute are parsed correctly,
// or not.

fn main() {
    let a = #![allow(warnings)] (1, 2);
    //~^ ERROR an inner attribute is not permitted in this context

    let b = (#![allow(warnings)] 1, 2);
    //~^ ERROR an inner attribute is not permitted in this context

    let c = {
        #![allow(warnings)]
        (#![allow(warnings)] 1, 2)
        //~^ ERROR an inner attribute is not permitted in this context
    };

    let d = {
        #![allow(warnings)]
        let e = (#![allow(warnings)] 1, 2);
        //~^ ERROR an inner attribute is not permitted in this context
        e
    };

    let e = [#![allow(warnings)] 1, 2];
    //~^ ERROR an inner attribute is not permitted in this context

    let f = [#![allow(warnings)] 1; 0];
    //~^ ERROR an inner attribute is not permitted in this context

    let g = match true { #![allow(warnings)] _ => {} };


    struct MyStruct { field: u8 }
    let h = MyStruct { #![allow(warnings)] field: 0 };
    //~^ ERROR an inner attribute is not permitted in this context
}
