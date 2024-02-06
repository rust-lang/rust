// Regression test for issue #119924.
// check-pass

pub struct Type;

trait Trait {
    fn provided() {
        impl Type {
            // This visibility qualifier used to get rejected.
            pub fn perform() {}
        }
    }
}

fn main() {}
