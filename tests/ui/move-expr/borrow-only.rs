//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let s = vec![1, 2, 3];
    let c = || {
        let t = &move(s);
        println!("{t:?}");
    };

    c();
}
