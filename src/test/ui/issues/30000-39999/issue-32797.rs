// check-pass

pub use bar::*;
mod bar {
    pub use super::*;
}

pub use baz::*;
mod baz {
    pub use main as f;
}

pub fn main() {}
