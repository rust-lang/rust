#![feature(plugin, collections)]

#![plugin(clippy)]
#![deny(clippy)]

extern crate collections;
use collections::linked_list::LinkedList;

pub fn test(foo: LinkedList<u8>) {  //~ ERROR I see you're using a LinkedList!
    println!("{:?}", foo)
}

fn main(){
    test(LinkedList::new());
}
