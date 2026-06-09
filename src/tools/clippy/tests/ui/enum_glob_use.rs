#![warn(clippy::enum_glob_use)]
#![allow(unused)]
#![warn(unused_imports)]

use std::cmp::Ordering::*;
//~^ enum_glob_use

enum Enum {
    Foo,
}

use self::Enum::*;
//~^ enum_glob_use

mod in_fn_test {
    fn blarg() {
        use crate::Enum::*;
        //~^ enum_glob_use

        let _ = Foo;
    }
}

mod blurg {
    #[allow(unused_imports)]
    pub use std::cmp::Ordering::*; // ok, re-export
}

fn main() {
    let _ = Foo;
    let _ = Less;
}
