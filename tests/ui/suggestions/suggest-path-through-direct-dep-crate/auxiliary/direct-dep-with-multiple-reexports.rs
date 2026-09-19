#![crate_type = "lib"]

extern crate transitive_dep;

mod private {
    pub use crate::transitive_dep::Struct;
}

#[doc(hidden)]
pub use crate::private::*;

#[doc(hidden)]
pub mod __private {
    pub use crate::private::*;
}
