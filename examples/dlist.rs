#![feature(plugin)]

#![plugin(clippy)]

extern crate collections;
use collections::linked_list::LinkedList;

pub fn test(foo: LinkedList<uint>) {
    println!("{:?}", foo)
}

fn main(){
    test(LinkedList::new());
}