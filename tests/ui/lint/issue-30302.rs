#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![deny(unreachable_patterns)]

enum Stack<T> {
    Nil,
    Cons(T, Box<Stack<T>>)
}

fn is_empty<T>(s: Stack<T>) -> bool {
    match s {
        Nil => true,
//~^ ERROR pattern binding `Nil` is named the same as one of the variants of the type `Stack`
        _ => false
//~^ ERROR unreachable pattern
    }
}

fn main() {}
