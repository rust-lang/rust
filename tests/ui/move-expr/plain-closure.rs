//@ run-pass
// This needs to remain a run-pass test to check capture behavior at runtime.
#![allow(incomplete_features)]
#![deny(unused_mut)]
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

    let data = vec![1];
    let mut f = || {
        let data = &mut move(data);
        data.push(2);
        data.len()
    };
    assert_eq!(f(), 2);
    assert_eq!(f(), 3);

    let mut f = || {
        let data = &mut move(Vec::<usize>::new());
        data.push(1);
        data.len()
    };
    assert_eq!(f(), 1);
    assert_eq!(f(), 2);

    let mut f = || move(0) += 1;
    f();

    let mut f = || move(String::new()) = String::from("updated");
    f();
    f();

    let data = String::from("raw");
    let mut f = || {
        let _ = &raw mut move(data);
    };
    f();
    f();

    let data = String::from("hello");
    let f = || (&move(data)).len();
    assert_eq!(f(), 5);
    assert_eq!(f(), 5);
}
