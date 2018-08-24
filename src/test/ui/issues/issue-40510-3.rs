#![feature(rustc_attrs)]
#![allow(unused)]

fn f() {
    let mut x: Vec<()> = Vec::new();

    || {
        || {
            x.push(())
        }
    };
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
