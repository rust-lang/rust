#![allow(unused)]

fn f() {
    let mut x: Vec<()> = Vec::new();

    || {
        || {
            x.push(())
        }
        //~^^^ ERROR captured variable cannot escape `FnMut` closure body
    };
}

fn main() {}
