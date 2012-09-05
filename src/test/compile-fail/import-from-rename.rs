// error-pattern:expected

use baz = foo::{bar};

mod foo {
    fn bar() {}
}

fn main() {
}