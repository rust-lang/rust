//@ edition: 2021
//@ aux-crate: reexport_source=reexport-source.rs

pub use reexport_source::Current as Item;
pub use reexport_source::redirected_module::Child;
