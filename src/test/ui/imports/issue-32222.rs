// check-pass

mod foo {
    pub fn bar() {}
}

pub use foo::*;
use b::bar;

mod foobar {
    use super::*;
}

mod a {
    pub mod bar {}
}

mod b {
    pub use a::bar;
}

fn main() {}
