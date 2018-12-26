// run-pass
#![allow(dead_code)]
#![allow(unused_imports)]
// This should resolve fine. Prior to fix, the last import
// was being tried too early, and marked as unrsolved before
// the glob import had a chance to be resolved.

mod bar {
    pub use self::middle::*;

    mod middle {
        pub use self::baz::Baz;

        mod baz {
            pub enum Baz {
                Baz1,
                Baz2
            }
        }
    }
}

mod foo {
    use bar::Baz::{Baz1, Baz2};
}

fn main() {}
