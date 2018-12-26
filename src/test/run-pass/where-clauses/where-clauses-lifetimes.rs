// run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

fn foo<'a, I>(mut it: I) where I: Iterator<Item=&'a isize> {}

fn main() {
    foo([1, 2].iter());
}
