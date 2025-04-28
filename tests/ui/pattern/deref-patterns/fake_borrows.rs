#![feature(deref_patterns)]
#![allow(incomplete_features)]

#[rustfmt::skip]
fn main() {
    let mut b = Box::new(false);
    match b {
        deref!(true) => {}
        _ if { *b = true; false } => {}
        //~^ ERROR cannot assign `*b` in match guard
        deref!(false) => {}
        _ => {},
    }
    match b {
        true => {}
        _ if { *b = true; false } => {}
        //~^ ERROR cannot assign `*b` in match guard
        false => {}
        _ => {},
    }
}
