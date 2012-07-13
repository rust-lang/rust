// error-pattern:expected

import baz = foo::{bar};

mod foo {
    fn bar() {}
}

fn main() {
}