// aux-build:impl_trait_aux.rs
// edition:2018

extern crate impl_trait_aux;

// @has impl_trait/fn.func.html
// @has - '//pre[@class="rust item-decl"]' "pub fn func<'a>(_x: impl Clone + Into<Vec<u8, Global>> + 'a)"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait_aux::func;

// @has impl_trait/fn.func2.html
// @has - '//pre[@class="rust item-decl"]' "func2<T>("
// @has - '//pre[@class="rust item-decl"]' "_x: impl Deref<Target = Option<T>> + Iterator<Item = T>,"
// @has - '//pre[@class="rust item-decl"]' "_y: impl Iterator<Item = u8> )"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait_aux::func2;

// @has impl_trait/fn.func3.html
// @has - '//pre[@class="rust item-decl"]' "func3("
// @has - '//pre[@class="rust item-decl"]' "_x: impl Iterator<Item = impl Iterator<Item = u8>> + Clone)"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait_aux::func3;

// @has impl_trait/fn.func4.html
// @has - '//pre[@class="rust item-decl"]' "func4<T>("
// @has - '//pre[@class="rust item-decl"]' "T: Iterator<Item = impl Clone>,"
pub use impl_trait_aux::func4;

// @has impl_trait/fn.func5.html
// @has - '//pre[@class="rust item-decl"]' "func5("
// @has - '//pre[@class="rust item-decl"]' "_f: impl for<'any> Fn(&'any str, &'any str) -> bool + for<'r> Other<T<'r> = ()>,"
// @has - '//pre[@class="rust item-decl"]' "_a: impl for<'alpha, 'beta> Auxiliary<'alpha, Item<'beta> = fn(_: &'beta ())>"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait_aux::func5;

// @has impl_trait/fn.async_fn.html
// @has - '//pre[@class="rust item-decl"]' "pub async fn async_fn()"
pub use impl_trait_aux::async_fn;

// @has impl_trait/struct.Foo.html
// @has - '//*[@id="method.method"]//h4[@class="code-header"]' "pub fn method<'a>(_x: impl Clone + Into<Vec<u8, Global>> + 'a)"
// @!has - '//*[@id="method.method"]//h4[@class="code-header"]' 'where'
pub use impl_trait_aux::Foo;

// @has impl_trait/struct.Bar.html
// @has - '//*[@id="method.async_foo"]' "pub async fn async_foo("
pub use impl_trait_aux::Bar;
