//@ edition:2018
//@ aux-crate:bar=bar.rs

pub const FOO: &str = "foo";
pub use bar::BAR;
