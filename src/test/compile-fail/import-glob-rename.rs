// error-pattern:expected

use baz = foo::*;

mod foo {
    #[legacy_exports];
    fn bar() {}
}

fn main() {
}