// compile-flags: -Z parse-only

// error-pattern:expected

use foo::{bar} as baz;

mod foo {
    pub fn bar() {}
}

fn main() {
}
