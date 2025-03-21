// Regression test for <https://github.com/rust-lang/rust/issues/107677>.

pub mod nested {
    //@ set foo_struct = "$.index[?(@.docs == 'Foo the struct')].id"

    /// Foo the struct
    pub struct Foo {}

    //@ set foo_fn = "$.index[?(@.docs == 'Foo the function')].id"

    #[allow(non_snake_case)]
    /// Foo the function
    pub fn Foo() {}
}

//@ ismany "$.index[?(@.inner.use.name == 'Foo')].inner.use.id" $foo_fn $foo_struct
//@ ismany "$.index[?(@.inner.use.name == 'Bar')].inner.use.id" $foo_fn $foo_struct

//@ count "$.index[?(@.inner.use.name == 'Foo')]" 2
//@ count "$.index[?(@.inner.use.name == 'Bar')]" 2
pub use Foo as Bar;
pub use nested::Foo;
