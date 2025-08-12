//@ edition:2021

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}
fn foo () -> impl FnMut() {
    let mut p = Point {x: 1, y: 2 };
    let mut c = || {
       p.x += 5;
       println!("{:?}", p);
        //~^ ERROR `p` does not live long enough
    };
    c
}
fn main() {
    let mut c = foo();
    c();
}
