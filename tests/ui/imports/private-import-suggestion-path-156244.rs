// PR #156244 comment
//@ revisions: edition_2015 edition_2018
//@[edition_2015] edition: 2015
//@[edition_2018] edition: 2018

mod a {
    pub struct One;
    pub struct Two;
}

mod b {
    use crate::a::{One, Two};
}

mod test {
    #[cfg(edition_2015)]
    use b::One;
    //[edition_2015]~^ ERROR struct import `One` is private [E0603]
    #[cfg(edition_2018)]
    use crate::b::One;
    //[edition_2018]~^ ERROR struct import `One` is private [E0603]
}

mod outer {
    pub mod actual {
        pub struct Item;
    }
}

mod rename {
    use crate::outer::actual as inner;
}

mod bad {
    use crate::b::{One, Two};
    //~^ ERROR struct import `One` is private [E0603]
    //~| ERROR struct import `Two` is private [E0603]
    use crate::rename::inner::Item as Item1;
    //~^ ERROR module import `inner` is private [E0603]
}

// Regression test for https://github.com/rust-lang/rust/issues/157455: no root `super`.
mod public {
    pub struct Hi;
}

mod testing {
    use super::public::Hi;
}

use testing::Hi;
//~^ ERROR struct import `Hi` is private [E0603]

// Regression test for https://github.com/rust-lang/rust/issues/157455: no private ancestors.
mod inaccessible_ancestor {
    mod private {
        pub mod public {
            pub struct Hi;
        }
    }

    pub mod testing {
        use super::private::public::Hi;
    }
}

use inaccessible_ancestor::testing::Hi;
//~^ ERROR struct import `Hi` is private [E0603]

// Regression test for https://github.com/rust-lang/rust/issues/157455: no external alias rewrite.
use std as s;

mod external_alias {
    use super::s::mem;
}

use external_alias::mem;
//~^ ERROR module import `mem` is private [E0603]

fn main() {}
