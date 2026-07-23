//@ edition: 2021
//@ aux-crate: reexport_source=reexport-source.rs

#![feature(edition_redirect)]

pub use reexport_source::Current as Item;
// The module itself is redirected, but its children are not. Canonical path
// resolution in an `edition_redirect` crate therefore finds `Child` in the real
// `redirected_module` and does not attach the module's redirect to this
// re-export.
pub use reexport_source::redirected_module::Child;
