//@ edition:2021

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}
fn main() {
    let mut p = Point {x: 1, y: 2 };

    let y = &mut p.y;
    let mut c = || {
    //~^ ERROR cannot borrow `p` as mutable more than once at a time
       let x = &mut p.x;
       println!("{:?}", p);
    };
    c();
    *y+=1;
}
