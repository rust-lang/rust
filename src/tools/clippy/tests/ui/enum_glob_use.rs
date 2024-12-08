#![warn(clippy::enum_glob_use)]
#![allow(unused)]
#![warn(unused_imports)]

use std::cmp::Ordering::*;

enum Enum {
    Foo,
}

use self::Enum::*;

mod in_fn_test {
    fn blarg() {
        use crate::Enum::*;

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
