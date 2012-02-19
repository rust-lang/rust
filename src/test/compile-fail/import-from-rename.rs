// error-pattern:expecting

import baz = foo::{bar};

mod foo {
    fn bar() {}
}

fn main() {
}