#![crate_name = "my_rand"]
#![deny(broken_intra_doc_links)]

pub trait RngCore {}
/// Rng extends [`RngCore`].
pub trait Rng: RngCore {}
