// aux-build:impl_trait_aux.rs

extern crate impl_trait_aux;

// @has impl_trait/fn.func.html
// @has - '//pre[@class="rust fn"]' "pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a)"
// @!has - '//pre[@class="rust fn"]' 'where'
pub use impl_trait_aux::func;

// @has impl_trait/struct.Foo.html
// @has - '//code[@id="method.v"]' "pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a)"
// @!has - '//code[@id="method.v"]' 'where'
pub use impl_trait_aux::Foo;
