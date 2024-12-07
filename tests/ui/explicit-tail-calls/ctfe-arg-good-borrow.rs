//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

pub const fn test(x: &Type) {
    const fn takes_borrow(_: &Type) {}

    become takes_borrow(x);
}

pub struct Type;

fn main() {}
