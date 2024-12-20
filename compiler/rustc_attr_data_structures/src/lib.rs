// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

mod attributes;
mod stability;
mod version;

pub use attributes::*;
pub(crate) use rustc_session::HashStableContext;
pub use stability::*;
pub use version::*;
