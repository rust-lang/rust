mod foo {
    pub struct Foo {
        bar: i32,
        baz: i32,
    }
}

use foo::*;

fn main() {
    let _f = Foo { bar: 0, quux: 0 }; //~ ERROR no field named `quux`
}
