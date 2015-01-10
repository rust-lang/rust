#![feature(plugin)]

#[plugin]
extern crate clippy;

extern crate collections;
use collections::dlist::DList;

pub fn test(foo: DList<uint>) {
    println!("{:?}", foo)
}

fn main(){
    test(DList::new());
}