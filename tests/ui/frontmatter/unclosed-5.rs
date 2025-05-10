----cargo
//~^ ERROR: unclosed frontmatter
//~| ERROR: frontmatters are experimental

// Similarly, a use statement should allow for recovery as well (as
// per unclosed-1.rs)

use std::env;

fn main() {}
