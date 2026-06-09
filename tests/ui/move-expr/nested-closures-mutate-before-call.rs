#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let mut x = String::from("hello");

    let outer = || {
        let inner = || move(x.clone());
        let y = inner();
        assert_eq!(y, "hello");
        assert_eq!(x, "hello");
    };

    x.push_str("more test"); //~ ERROR cannot borrow `x` as mutable because it is also borrowed as immutable

    outer();
}
