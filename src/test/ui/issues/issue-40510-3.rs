// compile-pass
#![allow(unused)]

fn f() {
    let mut x: Vec<()> = Vec::new();

    || {
        || {
            x.push(())
        }
    };
}


fn main() {}
