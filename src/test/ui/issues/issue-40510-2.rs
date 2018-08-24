#![feature(rustc_attrs)]
#![allow(unused)]

fn f() {
    let x: Box<()> = Box::new(());

    || {
        &x
    };
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
