// error-pattern:can't rename import list

import baz = foo::{bar};

mod foo {
    fn bar() {}
}

fn main() {
}