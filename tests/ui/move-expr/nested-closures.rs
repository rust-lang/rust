//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let x = String::from("hello");
    let outer = || {
        let inner = || move(x.clone());
        let y = inner();
        assert_eq!(y, "hello");
        assert_eq!(x, "hello");
    };

    outer();
}
