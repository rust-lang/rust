// aux-build:impl_trait_aux.rs

extern crate impl_trait_aux;

// @has impl_trait/fn.func.html
// @has - '//pre[@class="rust fn"]' '(_x: impl '
// @has - '//pre[@class="rust fn"]' 'Clone'
// @has - '//pre[@class="rust fn"]' 'Into'
// @has - '//pre[@class="rust fn"]' "'a"
// @!has - '//pre[@class="rust fn"]' 'where'
pub use impl_trait_aux::func;

// @has impl_trait/struct.Foo.html
// @has - '//code[@id="method.v"]' '(_x: impl '
// @has - '//code[@id="method.v"]' 'Clone'
// @has - '//code[@id="method.v"]' 'Into'
// @has - '//code[@id="method.v"]' "'a"
// @!has - '//code[@id="method.v"]' 'where'
pub use impl_trait_aux::Foo;
