#![deny(broken_intra_doc_links)]

/// [somewhere]
//~^ ERROR unresolved link to `somewhere`
pub use std::str::*;
/// [aloha]
//~^ ERROR unresolved link to `aloha`
pub use std::task::RawWakerVTable;
