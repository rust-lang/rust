use foo::*;

mod foo {
    pub mod bar {
        pub mod bar {
            pub mod bar {}
        }
    }
}

use bar::bar; //~ ERROR `bar` is ambiguous
use bar::*;

fn main() { }
