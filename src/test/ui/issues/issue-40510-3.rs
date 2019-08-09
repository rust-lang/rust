#![feature(rustc_attrs)]
#![allow(unused)]

// revisions: migrate nll
#![cfg_attr(nll, feature(nll))]

fn f() {
    let mut x: Vec<()> = Vec::new();

    || {
        || {
            x.push(())
        }
        //[migrate]~^^^ WARNING captured variable cannot escape `FnMut` closure body
        //[migrate]~| WARNING this error has been downgraded to a warning
        //[migrate]~| WARNING this warning will become a hard error in the future
        //[nll]~^^^^^^ ERROR captured variable cannot escape `FnMut` closure body
    };
}

#[rustc_error]
fn main() {}
//[migrate]~^ ERROR
