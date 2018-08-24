#![feature(rustc_attrs)]
#![allow(unused)]

macro_rules! make_item {
    () => { fn f() {} }
}

macro_rules! make_stmt {
    () => { let x = 0; }
}

fn f() {
    make_item! {}
}

fn g() {
    make_stmt! {}
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
