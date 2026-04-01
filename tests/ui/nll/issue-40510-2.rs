//@ check-pass
#![allow(unused)]

fn f() {
    let x: Box<()> = Box::new(());

    || {
        &x
    };
}


fn main() {}
