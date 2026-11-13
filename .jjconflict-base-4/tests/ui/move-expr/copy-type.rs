//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let x = 22;
    let c = || {
        let y = move(x);
        let z = x;
        assert_eq!(y + z, 44);
    };

    c();
    c();
}
