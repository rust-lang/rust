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
}
