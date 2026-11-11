// Ensure we don't misclassify `impl Trait` as TAIT/ATPIT if located inside an anon const in a
// type alias/assoc type.
// issue: <https://github.com/rust-lang/rust/issues/139055>
//@ check-pass
#![forbid(unstable_features)]

struct Girder<const N: usize>;

type Alias = Girder<{
    fn pass(input: impl Sized) -> impl Sized { input }
    0
}>;

trait Trait {
    type Assoc;
}

impl Trait for () {
    type Assoc = [(); {
        fn pass(input: impl Sized) -> impl Sized { input }
        0
    }];
}

fn main() {}
