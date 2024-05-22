#![allow(unused)]

fn f() {
    let mut x: Box<()> = Box::new(());

    || {
        &mut x
    };
    //~^^ ERROR captured variable cannot escape `FnMut` closure body
}

fn main() {}
