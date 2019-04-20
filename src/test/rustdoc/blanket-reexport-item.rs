#![crate_name = "foo"]

// @has foo/struct.S.html '//h3[@id="impl-Into%3CU%3E"]//code' 'impl<T, U> Into<U> for T'
pub struct S2 {}
mod m {
    pub struct S {}
}
pub use m::*;
