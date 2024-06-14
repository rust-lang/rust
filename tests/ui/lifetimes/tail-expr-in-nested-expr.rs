//@ edition: 2024
//@ compile-flags: -Zunstable-options

#![feature(shorter_tail_lifetimes)]

fn main() {
    let _ = { String::new().as_str() }.len();
    //~^ ERROR temporary value dropped while borrowed
}
