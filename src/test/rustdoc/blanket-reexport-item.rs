#![crate_name = "foo"]

// @has foo/struct.S.html '//h3[@id="impl-Into"]//code' 'impl<T, U> Into for T'
pub struct S2 {}
mod m {
    pub struct S {}
}
pub use m::*;
