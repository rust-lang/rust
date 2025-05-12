// Test that we can use ! as an argument to a trait impl.

//@ check-pass

#![feature(never_type)]

struct Wub;

impl PartialEq<!> for Wub {
    fn eq(&self, other: &!) -> bool {
        *other
    }
}

fn main() {
    let _ = Wub == panic!("oh no!");
}
