#![allow(unused)]

fn f() {
    let mut x: Vec<()> = Vec::new();

    || {
        || { //~ ERROR captured variable cannot escape `FnMut` closure body
            x.push(())
        }
    };
}


fn main() {}
