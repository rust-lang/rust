// error-pattern:expected

use baz = foo::*;

mod foo {
    fn bar() {}
}

fn main() {
}