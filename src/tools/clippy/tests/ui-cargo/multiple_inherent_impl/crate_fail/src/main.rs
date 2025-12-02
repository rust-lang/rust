#![allow(dead_code)]
#![deny(clippy::multiple_inherent_impl)]

struct S;
struct T;

impl S {
    fn first() {}
}

impl S {
    //^ Must trigger
    fn second() {}
}

impl T {
    fn first() {}
}

mod a {
    use super::T;
    impl T {
        //^ Must trigger
        fn second() {}
    }
}

mod b;

impl b::S {
    //^ Must NOT trigger
    fn first() {}
    fn second() {}
}

impl b::T {
    //^ Must trigger
    fn second() {}
}

fn main() {}
