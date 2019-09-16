#![feature(stmt_expr_attributes)]

// Test that various placements of the inner attribute are parsed correctly,
// or not.

fn main() {
    let a = #![allow(warnings)] (1, 2);
    //~^ ERROR an inner attribute is not permitted in this context

    let b = (#![allow(warnings)] 1, 2);

    let c = {
        #![allow(warnings)]
        (#![allow(warnings)] 1, 2)
    };

    let d = {
        #![allow(warnings)]
        let e = (#![allow(warnings)] 1, 2);
        e
    };
}
