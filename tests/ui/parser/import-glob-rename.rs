//@error-in-other-file:expected

use foo::* as baz;

mod foo {
    pub fn bar() {}
}

fn main() {
}
