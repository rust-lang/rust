//@ check-pass

// similar `cycle-import-in-diff-module-0.rs`

mod a {
    pub(crate) use crate::s;
}
mod b {
    pub mod s {}
}
use self::b::*;
use self::a::s;

fn main() {}
