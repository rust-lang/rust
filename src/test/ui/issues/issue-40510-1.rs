#![feature(rustc_attrs)]
#![allow(unused)]

fn f() {
    let mut x: Box<()> = Box::new(());

    || {
        &mut x
    };
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
