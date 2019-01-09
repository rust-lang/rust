// compile-pass
// skip-codegen
#![allow(dead_code)]
pub type T = ();
mod foo { pub use super::T; }
mod bar { pub use super::T; }

pub use foo::*;
pub use bar::*;

mod baz {
    pub type T = ();
    mod foo { pub use super::T as S; }
    mod bar { pub use super::foo::S as T; }
    pub use self::bar::*;
}


fn main() {}
