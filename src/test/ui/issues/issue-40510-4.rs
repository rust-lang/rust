#![feature(rustc_attrs)]
#![allow(unused)]

fn f() {
    let x: Vec<()> = Vec::new();

    || {
        || {
            x.len()
        }
    };
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
