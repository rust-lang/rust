// check-pass

mod foo {
    pub use bar::*;
    pub use main as f;
}

mod bar {
    pub use foo::*;
}

pub use foo::*;
pub use baz::*;
mod baz {
    pub use super::*;
}

pub fn main() {}
