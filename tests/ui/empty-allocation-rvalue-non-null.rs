//@ run-pass

#![allow(unused_variables)]

pub fn main() {
    let x: () = *Box::new(());
}
