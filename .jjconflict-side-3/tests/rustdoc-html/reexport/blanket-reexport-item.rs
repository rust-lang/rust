#![crate_name = "foo"]

//@ has foo/struct.S.html '//*[@id="impl-Into%3CU%3E-for-T"]//h3[@class="code-header"]' 'impl<T, U> Into<U> for T'
pub struct S2 {}
mod m {
    pub struct S {}
}
pub use m::*;
