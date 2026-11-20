//@ run-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::cell::Cell;

fn main() {
    let created = Cell::new(0);
    let c = || {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        n
    };
    assert_eq!(created.get(), 1);
    assert_eq!(c(), 1);
    assert_eq!(c(), 1);
    assert_eq!(created.get(), 1);

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
