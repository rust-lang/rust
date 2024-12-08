//@ edition:2021
//@ run-pass

// Test whether if we can do precise capture when using nested clsoure.

struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut p = Point { x: 5, y: 20 };

    // c1 should capture `p.x` via immutable borrow and
    // `p.y` via mutable borrow.
    let mut c1 = || {
        println!("{}", p.x);

        let incr = 10;

        let mut c2 = || p.y += incr;
        c2();

        println!("{}", p.y);
    };

    c1();

    // This should not throw an error because `p.x` is borrowed via Immutable borrow,
    // and multiple immutable borrow of the same place are allowed.
    let px = &p.x;

    println!("{}", px);

    c1();
}
