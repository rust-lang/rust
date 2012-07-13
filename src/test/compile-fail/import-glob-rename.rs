// error-pattern:expected

import baz = foo::*;

mod foo {
    fn bar() {}
}

fn main() {
}