//! Definitions shared by macros / syntax extensions and e.g. `rustc_middle`.

use rustc_macros::{Decodable, Encodable, HashStable_Generic};

pub mod allocator;
pub mod autodiff_attrs;
pub mod typetree;
