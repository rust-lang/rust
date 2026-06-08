//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let s = String::from("hello");
    let c = || {
        let t = move(s);
        println!("{}", t.len());
    };
    c();

    let a = String::from("hello");
    let b = String::from("world");
    let c = || {
        let x = move(a);
        let y = move(b);
        println!("{} {}", x, y);
    };
    c();

}
