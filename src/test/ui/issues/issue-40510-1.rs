#![allow(unused)]

fn f() {
    let mut x: Box<()> = Box::new(());

    || {
        &mut x //~ ERROR cannot infer
    };
}


fn main() {}
