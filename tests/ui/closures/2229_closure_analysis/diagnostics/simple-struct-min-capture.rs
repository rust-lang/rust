//@ edition:2021

// Test that borrow checker error is accurate and that min capture pass of the
// closure analysis is working as expected.

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut p = Point { x: 10, y: 20 };

    // `p` is captured via mutable borrow.
    let mut c = || {
        p.x += 10;
        println!("{:?}", p);
    };


    println!("{:?}", p);
    //~^ ERROR: cannot borrow `p` as immutable because it is also borrowed as mutable
    c();
}
