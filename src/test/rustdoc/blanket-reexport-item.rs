#![crate_name = "foo"]

// @has foo/struct.S.html '//*[@id="impl-Into%3CU%3E"]//h3[@class="code-header in-band"]' 'impl<T, U> Into<U> for T'
pub struct S2 {}
mod m {
    pub struct S {}
}
pub use m::*;
