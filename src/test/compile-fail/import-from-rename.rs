// error-pattern:expected

use baz = foo::{bar};

mod foo {
    #[legacy_exports];
    fn bar() {}
}

fn main() {
}