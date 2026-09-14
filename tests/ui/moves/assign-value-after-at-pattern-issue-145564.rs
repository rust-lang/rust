//@ run-rustfix
#![allow(unused_variables)]

fn main() {
    let ref mut x @ _v;
    *x = 1; //~ ERROR used binding `x` isn't initialized

    let a @ _b: i32;
    println!("{}", a); //~ ERROR used binding `a` isn't initialized

    let ref c @ _d: i32;
    println!("{:?}", c); //~ ERROR used binding `c` isn't initialized

    let ref e: i32;
    println!("{:?}", e); //~ ERROR used binding `e` isn't initialized

    let ref mut y;
    *y = 1; //~ ERROR used binding `y` isn't initialized
}
