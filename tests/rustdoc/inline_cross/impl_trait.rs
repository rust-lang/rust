// aux-build:impl_trait_aux.rs
// edition:2018

extern crate impl_trait_aux;

// @has impl_trait/fn.func.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "pub fn func<'a>(_x: impl Clone + Into<Vec<u8, Global>> + 'a)"
// @!has - '//div[@class="item-decl"]/pre[@class="rust"]' 'where'
pub use impl_trait_aux::func;

// @has impl_trait/fn.func2.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "func2<T>("
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "_x: impl Deref<Target = Option<T>> + Iterator<Item = T>,"
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "_y: impl Iterator<Item = u8>)"
// @!has - '//div[@class="item-decl"]/pre[@class="rust"]' 'where'
pub use impl_trait_aux::func2;

// @has impl_trait/fn.func3.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "func3("
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "_x: impl Iterator<Item = impl Iterator<Item = u8>> + Clone)"
// @!has - '//div[@class="item-decl"]/pre[@class="rust"]' 'where'
pub use impl_trait_aux::func3;

// @has impl_trait/fn.func4.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "func4<T>("
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "T: Iterator<Item = impl Clone>,"
pub use impl_trait_aux::func4;

// @has impl_trait/fn.func5.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "func5("
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "_f: impl for<'any> Fn(&'any str, &'any str) -> bool + for<'r> Other<T<'r> = ()>,"
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "_a: impl for<'alpha, 'beta> Auxiliary<'alpha, Item<'beta> = fn(_: &'beta ())>"
// @!has - '//div[@class="item-decl"]/pre[@class="rust"]' 'where'
pub use impl_trait_aux::func5;

// @has impl_trait/fn.async_fn.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "pub async fn async_fn()"
pub use impl_trait_aux::async_fn;

// @has impl_trait/struct.Foo.html
// @has - '//*[@id="method.method"]//h4[@class="code-header"]' "pub fn method<'a>(_x: impl Clone + Into<Vec<u8, Global>> + 'a)"
// @!has - '//*[@id="method.method"]//h4[@class="code-header"]' 'where'
pub use impl_trait_aux::Foo;

// @has impl_trait/struct.Bar.html
// @has - '//*[@id="method.async_foo"]' "pub async fn async_foo("
pub use impl_trait_aux::Bar;
