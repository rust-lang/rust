#![feature(box_syntax)]

fn main() {
    let x: Box<_> = box 5;
    let y = x;
    println!("{}", *x); //~ ERROR use of moved value: `*x`
    y.clone();
}
