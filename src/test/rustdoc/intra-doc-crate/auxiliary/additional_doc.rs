#![crate_name = "my_rand"]
#![deny(intra_doc_link_resolution_failure)]

pub trait RngCore {}
/// Rng extends [`RngCore`].
pub trait Rng: RngCore {}
