// Regression test for #121607 and for part of issue #119924.
//@ check-pass

trait Trait {
    fn provided() {
        pub struct Type;

        impl Type {
            // This visibility qualifier used to get rejected.
            pub fn perform() {}
        }
    }
}

fn main() {}
