// Regression test for issue #89342 and for part of #119924.
//@ check-pass

struct Expr<const N: u32>;

trait Trait0 {
    fn required(_: Expr<{
        struct Type;

        impl Type {
            // This visibility qualifier used to get rejected.
            pub fn perform() {}
        }

        0
    }>);
}

trait Trait1 {}

impl Trait1 for ()
where
    [(); {
        struct Type;

        impl Type {
            // This visibility qualifier used to get rejected.
            pub const STORE: Self = Self;
        }

        0
    }]:
{}

fn main() {}
