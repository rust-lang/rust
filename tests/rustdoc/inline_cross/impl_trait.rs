// aux-crate:impl_trait=impl_trait.rs
// edition:2018
#![crate_name = "user"]

// @has user/fn.func.html
// @has - '//pre[@class="rust item-decl"]' "pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a)"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait::func;

// @has user/fn.func2.html
// @has - '//pre[@class="rust item-decl"]' "func2<T>("
// @has - '//pre[@class="rust item-decl"]' "_x: impl Deref<Target = Option<T>> + Iterator<Item = T>,"
// @has - '//pre[@class="rust item-decl"]' "_y: impl Iterator<Item = u8> )"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait::func2;

// @has user/fn.func3.html
// @has - '//pre[@class="rust item-decl"]' "func3("
// @has - '//pre[@class="rust item-decl"]' "_x: impl Iterator<Item = impl Iterator<Item = u8>> + Clone)"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait::func3;

// @has user/fn.func4.html
// @has - '//pre[@class="rust item-decl"]' "func4<T>("
// @has - '//pre[@class="rust item-decl"]' "T: Iterator<Item = impl Clone>,"
pub use impl_trait::func4;

// @has user/fn.func5.html
// @has - '//pre[@class="rust item-decl"]' "func5("
// @has - '//pre[@class="rust item-decl"]' "_f: impl for<'any> Fn(&'any str, &'any str) -> bool + for<'r> Other<T<'r> = ()>,"
// @has - '//pre[@class="rust item-decl"]' "_a: impl for<'beta, 'alpha, '_gamma> Auxiliary<'alpha, Item<'beta> = fn(_: &'beta ())>"
// @!has - '//pre[@class="rust item-decl"]' 'where'
pub use impl_trait::func5;

// @has user/struct.Foo.html
// @has - '//*[@id="method.method"]//h4[@class="code-header"]' "pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a)"
// @!has - '//*[@id="method.method"]//h4[@class="code-header"]' 'where'
pub use impl_trait::Foo;

// @has user/fn.rpit_fn.html
// @has - '//pre[@class="rust item-decl"]' "rpit_fn() -> impl Fn() -> bool"
pub use impl_trait::rpit_fn;

// @has user/fn.rpit_fn_mut.html
// @has - '//pre[@class="rust item-decl"]' "rpit_fn_mut() -> impl for<'a> FnMut(&'a str) -> &'a str"
pub use impl_trait::rpit_fn_mut;
