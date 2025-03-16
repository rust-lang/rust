//@ check-pass
//@ edition:2024

#![allow(irrefutable_let_patterns)]
fn main() {
    let _a = 0..1;

    if let x = (0..1) {
        eprintln!("x: {:?}", x);
    }
    if let x = (0..1) &&
        let _y = (0..2)
    {
        eprintln!("x: {:?}", x);
    }
}
