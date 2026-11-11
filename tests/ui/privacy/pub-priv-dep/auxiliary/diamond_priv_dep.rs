//@ aux-crate:shared=shared.rs

extern crate shared;

pub use shared::Shared;

pub struct SharedInType {
    pub f: Shared
}
