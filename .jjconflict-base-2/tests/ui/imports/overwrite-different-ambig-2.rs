mod m1 {
    mod inner {
        pub struct S {}
    }
    pub use self::inner::*;

    #[derive(Debug)]
    pub struct S {}
}

mod m2 {
    pub struct S {}
}

// First we have a glob ambiguity in this glob (with `m2::*`).
// Then we re-fetch `m1::*` because non-glob `m1::S` materializes from derive,
// and we need to make sure that the glob ambiguity is not lost during re-fetching.
use m1::*;
use m2::*;

fn main() {
    let _: m1::S = S {}; //~ ERROR `S` is ambiguous
                         //~| WARN this was previously accepted
}
