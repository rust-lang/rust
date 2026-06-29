//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let v = "Hello, Ferris".to_string();
    let r = || {
        || (move(move(v.clone()))).len()
    };

    assert_eq!(r()(), v.len());
}
