use foo::{bar} as baz; //~ ERROR expected `;`, found keyword `as`

mod foo {
    pub fn bar() {}
}

fn main() {
}
