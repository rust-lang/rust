// error-pattern:expected

use foo::* as baz;

mod foo {
    pub fn bar() {}
}

fn main() {
}
