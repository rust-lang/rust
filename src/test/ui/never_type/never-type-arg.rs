// Test that we can use ! as an argument to a trait impl.

// check-pass

struct Wub;

impl PartialEq<!> for Wub {
    fn eq(&self, other: &!) -> bool {
        *other
    }
}

fn main() {
    let _ = Wub == panic!("oh no!");
}
