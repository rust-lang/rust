// Regression test for #81059.
// edition:2024
//@ run-rustfix

#![allow(unused)]

fn test(outer: &mut Option<i32>) {
    //~^ NOTE: the binding is already a mutable borrow
    //~| HELP: consider making the binding mutable if you need to reborrow multiple times
    match (&mut outer, 23) {
        //~^ ERROR: cannot borrow `outer` as mutable, as it is not declared as mutable
        //~| NOTE: cannot borrow as mutable
        //~| HELP: to reborrow the mutable reference, add `*`
        (Some(inner), _) => {
            *inner = 17;
        }
        _ => {
            *outer = Some(2);
        }
    }
}

fn main() {}
