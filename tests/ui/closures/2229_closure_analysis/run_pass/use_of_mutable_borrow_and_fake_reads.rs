//@ edition:2021
//@check-pass
#![feature(rustc_attrs)]

fn main() {
    let mut x = 0;
    let c = || {
        &mut x; // mutable borrow of `x`
        match x { _ => () } // fake read of `x`
    };
}
