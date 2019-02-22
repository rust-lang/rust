// compile-pass

use crate::foo::Bar; //~ WARNING first import

mod foo {
    pub type Bar = i32;
}

fn baz() -> Bar {
    3
}

fn main() {
    use crate::foo::Bar; //~ WARNING redundant import
    let _a: Bar = 3;
    baz();
}
