//@ edition:2021
//@ run-pass

// Test that we can immutably borrow field of an instance of a structure from within a closure,
// while having a mutable borrow to another field of the same instance outside the closure.

struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut p = Point { x: 10, y: 10 };

    let c = || {
        println!("{}", p.x);
    };

    // `c` should only capture `p.x`, therefore mutating `p.y` is allowed.
    let py = &mut p.y;

    c();
    *py = 20;
}
