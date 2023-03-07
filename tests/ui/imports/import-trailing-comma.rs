// run-pass
// pretty-expanded FIXME #23616

use foo::bar::{baz, quux,};

mod foo {
    pub mod bar {
        pub fn baz() { }
        pub fn quux() { }
    }
}

pub fn main() { baz(); quux(); }
