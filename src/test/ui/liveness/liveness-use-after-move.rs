#![feature(box_syntax)]

fn main() {
    let x: Box<_> = box 5;
    let y = x;
    println!("{}", *x); //~ ERROR borrow of moved value: `x`
    y.clone();
}
