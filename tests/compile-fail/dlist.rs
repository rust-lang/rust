#![feature(plugin, collections)]

#![plugin(clippy)]
#![deny(clippy)]

extern crate collections;
use collections::linked_list::LinkedList;

pub fn test(_: LinkedList<u8>) {  //~ ERROR I see you're using a LinkedList!
}

fn main(){
    test(LinkedList::new()); //~ ERROR I see you're using a LinkedList!
}
