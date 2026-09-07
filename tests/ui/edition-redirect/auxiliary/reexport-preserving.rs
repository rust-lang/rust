//@ edition: 2021
//@ aux-crate: reexport_source=reexport-source.rs

pub use reexport_source::Current as Item;
// Redirects are selected when this crate first imports the external name, so
// both re-exports are fixed according to this crate's edition.
pub use reexport_source::redirected_module::Child;
