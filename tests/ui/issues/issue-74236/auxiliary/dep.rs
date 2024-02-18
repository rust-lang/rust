//@ edition:2018

mod private { pub struct Pub; }

// Reexport built-in attribute without a DefId (requires Rust 2018).
pub use cfg_attr as attr;
// This export needs to be after the built-in attribute to trigger the bug.
pub use private::Pub as Renamed;
