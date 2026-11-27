//@ check-pass

#![feature(return_type_notation)]
#![allow(dead_code)]

struct A<const B: usize>;

trait C {
    fn d(&self) -> impl C<d(..):>;
}

fn main() {}
