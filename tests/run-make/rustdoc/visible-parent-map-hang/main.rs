// #160439: nested #[doc(hidden)] modules made rustdoc hang here.
#![crate_type = "lib"]

use std::fmt;

pub use dep::Tr;

pub struct Local;
impl dep::Tr for Local {
    fn x() {}
}
impl dep::Tr2 for Local {
    fn x() {}
}
impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x")
    }
}
impl Default for Local {
    fn default() -> Self {
        Local
    }
}
