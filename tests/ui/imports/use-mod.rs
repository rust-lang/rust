// run-pass

#![allow(unused_imports)]
// pretty-expanded FIXME #23616

pub use foo::bar::{self, First};
use self::bar::Second;

mod foo {
    pub use self::bar::baz::{self};

    pub mod bar {
        pub mod baz {
            pub struct Fourth;
        }
        pub struct First;
        pub struct Second;
    }

    pub struct Third;
}

mod baz {
    use super::foo::{bar, self};
    pub use foo::Third;
}

fn main() {
    let _ = First;
    let _ = Second;
    let _ = baz::Third;
    let _ = foo::baz::Fourth;
}
