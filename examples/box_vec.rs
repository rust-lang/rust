#![feature(phase)]

#[phase(plugin)]
extern crate clippy;

pub fn test(foo: Box<Vec<uint>>) {
    println!("{}", foo)
}

fn main(){
    test(box Vec::new());
}