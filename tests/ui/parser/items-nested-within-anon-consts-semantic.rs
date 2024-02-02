// Regression test for issues #119924 and #89342.
// check-pass

struct Type;
struct Expr<const N: u32>;

trait Trait0 {
    fn required(_: Expr<{
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
        impl Type {
            // This visibility qualifier used to get rejected.
            pub const STORE: Self = Self;
        }
        0
    }]:
{}

trait Trait2<const B: bool> {}

fn f(_: impl Trait2<{
    // This impl-Trait used to get rejected as "nested" impl-Trait.
    fn g(_: impl Sized) {}
    false
}>) {}

fn scope() {
    let _: <Parametrized<{
        // This impl-Trait used get rejected with "not allowed in path parameters".
        fn run(_: impl Sized) {}
        0
    }> as Trait3>::Projected;
}

trait Trait3 { type Projected; }
struct Parametrized<const N: usize>;

impl<const N: usize> Trait3 for Parametrized<N> { type Projected = (); }

fn main() {}
