// Prefix in imports with empty braces should be resolved and checked privacy, stability, etc.

mod m {
    mod n {}
}

use m::n::{};
//~^ ERROR module `n` is private

fn main() {}
