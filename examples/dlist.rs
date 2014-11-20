#![feature(phase)]

#[phase(plugin)]
extern crate rust_clippy;
extern crate collections;
use collections::dlist::DList;

pub fn test(foo: DList<uint>) {
    println!("{}", foo)
}

fn main(){
    test(DList::new());
}