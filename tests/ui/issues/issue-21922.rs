//@ run-pass
use std::ops::Add;
fn show(z: i32) {
    println!("{}", z)
}
fn main() {
    let x = 23;
    let y = 42;
    show(Add::add( x,  y));
    show(Add::add( x, &y));
    show(Add::add(&x,  y));
    show(Add::add(&x, &y));
    show( x +  y);
    show( x + &y);
    show(&x +  y);
    show(&x + &y);
}
