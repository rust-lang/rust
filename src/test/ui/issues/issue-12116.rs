#![feature(box_patterns)]
#![feature(box_syntax)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(unreachable_patterns)]

enum IntList {
    Cons(isize, Box<IntList>),
    Nil
}

fn tail(source_list: &IntList) -> IntList {
    match source_list {
        &IntList::Cons(val, box ref next_list) => tail(next_list),
        &IntList::Cons(val, box IntList::Nil)  => IntList::Cons(val, box IntList::Nil),
//~^ ERROR unreachable pattern
        _                          => panic!()
    }
}

fn main() {}
