// Minimized case from #62767.
mod m {
    pub enum Same {
        Same,
    }
}

use m::*;

// The variant `Same` introduced by this import is also considered when resolving the prefix
// `Same::` during import validation to avoid effects similar to time travel (#74556).
use Same::Same; //~ ERROR unresolved import `Same`

// Case from #74556.
mod foo {
    pub mod bar {
        pub mod bar {
            pub fn foobar() {}
        }
    }
}

use foo::*;
use bar::bar; //~ ERROR unresolved import `bar::bar`
              //~| ERROR inconsistent resolution for an import
use bar::foobar;

fn main() {}
