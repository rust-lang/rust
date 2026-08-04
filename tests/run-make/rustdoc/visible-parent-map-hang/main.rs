#![crate_type = "lib"]
extern crate dep;
pub use dep::Tr;
use std::fmt;

pub struct Local;
impl dep::Tr for Local { fn x() {} }
impl dep::Tr2 for Local { fn x() {} }
impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "x") }
}
impl Default for Local { fn default() -> Self { Local } }

