// error-pattern:globbed imports can't be renamed

import baz = foo::*;

mod foo {
    fn bar() {}
}

fn main() {
}