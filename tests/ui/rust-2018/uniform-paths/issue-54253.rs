// edition:2018

// Dummy import that previously introduced uniform path canaries.
use std;

// fn version() -> &'static str {""}

mod foo {
    // Error wasn't reported, despite `version` being commented out above.
    use crate::version; //~ ERROR unresolved import `crate::version`

    fn bar() {
        version();
    }
}

fn main() {}
