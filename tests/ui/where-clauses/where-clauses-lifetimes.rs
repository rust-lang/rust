//@ run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]

fn foo<'a, I>(mut it: I) where I: Iterator<Item=&'a isize> {}

fn main() {
    foo([1, 2].iter());
}
