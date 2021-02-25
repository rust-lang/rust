#![crate_name = "inner"]
#![feature(external_doc)]
#[doc(include = "docs.md")]
pub struct HasDocInclude {}
pub struct NoDocs {}
