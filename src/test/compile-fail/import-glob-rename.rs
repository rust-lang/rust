// error-pattern:expecting

import baz = foo::*;

mod foo {
    fn bar() {}
}

fn main() {
}