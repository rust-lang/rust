// edition:2021

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}
fn foo () -> impl FnMut()->() {
    let mut p = Point {x: 1, y: 2 };
    let mut c = || {
    //~^ ERROR closure may outlive the current function, but it borrows `p`
       p.x+=5;
       println!("{:?}", p);
    };
    c
}
fn main() {
    let c = foo();
    c();
}
