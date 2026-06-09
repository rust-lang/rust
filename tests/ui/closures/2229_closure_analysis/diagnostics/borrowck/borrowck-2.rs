//@ edition:2021

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}
fn main() {
    let mut p = Point {x: 1, y: 2 };

    let y = &p.y;
    let mut c = || {
    //~^ ERROR cannot borrow `p` as mutable because it is also borrowed as immutable
       println!("{:?}", p);
       let x = &mut p.x;
    };
    c();
    println!("{}", y);
}
