// aux-build:impl_trait_aux.rs
// edition:2018
// ignore-tidy-linelength

extern crate impl_trait_aux;

// @has impl_trait/fn.func.html
// @has - '//pre[@class="rust fn"]' "pub fn func<'a>(_x: impl Clone + Into<Vec<u8, Global>> + 'a)"
// @!has - '//pre[@class="rust fn"]' 'where'
pub use impl_trait_aux::func;

// @has impl_trait/fn.func2.html
// @has - '//pre[@class="rust fn"]' "func2<T>("
// @has - '//pre[@class="rust fn"]' "_x: impl Deref<Target = Option<T>> + Iterator<Item = T>,"
// @has - '//pre[@class="rust fn"]' "_y: impl Iterator<Item = u8>)"
// @!has - '//pre[@class="rust fn"]' 'where'
pub use impl_trait_aux::func2;

// @has impl_trait/fn.func3.html
// @has - '//pre[@class="rust fn"]' "func3("
// @has - '//pre[@class="rust fn"]' "_x: impl Clone + Iterator<Item = impl Iterator<Item = u8>>)"
// @!has - '//pre[@class="rust fn"]' 'where'
pub use impl_trait_aux::func3;

// @has impl_trait/fn.func4.html
// @has - '//pre[@class="rust fn"]' "func4<T>("
// @has - '//pre[@class="rust fn"]' "T: Iterator<Item = impl Clone>,"
pub use impl_trait_aux::func4;

// @has impl_trait/fn.async_fn.html
// @has - '//pre[@class="rust fn"]' "pub async fn async_fn()"
pub use impl_trait_aux::async_fn;

// @has impl_trait/struct.Foo.html
// @has - '//*[@id="method.method"]//code' "pub fn method<'a>(_x: impl Clone + Into<Vec<u8, Global>> + 'a)"
// @!has - '//*[@id="method.method"]//code' 'where'
pub use impl_trait_aux::Foo;

// @has impl_trait/struct.Bar.html
// @has - '//*[@id="method.async_foo"]' "pub async fn async_foo("
pub use impl_trait_aux::Bar;
