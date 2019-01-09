// compile-pass
#![allow(unused)]

fn f() {
    let mut x: Box<()> = Box::new(());

    || {
        &mut x
    };
}


fn main() {}
