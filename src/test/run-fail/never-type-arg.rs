// Test that we can use ! as an argument to a trait impl.

// error-pattern:oh no!

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
