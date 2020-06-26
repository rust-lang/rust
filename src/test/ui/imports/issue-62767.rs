// check-pass

mod m {
    pub enum Same {
        Same,
    }
}

use m::*;

// The variant `Same` introduced by this import is not considered when resolving the prefix
// `Same::` during import validation (issue #62767).
use Same::Same;

fn main() {}
