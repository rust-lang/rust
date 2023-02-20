// Regression test for <https://github.com/rust-lang/rust/issues/107677>.

#![feature(no_core)]
#![no_core]

pub mod nested {
    // @set foo_struct = "$.index[*][?(@.docs == 'Foo the struct')].id"

    /// Foo the struct
    pub struct Foo {}

    // @set foo_fn = "$.index[*][?(@.docs == 'Foo the function')].id"

    #[allow(non_snake_case)]
    /// Foo the function
    pub fn Foo() {}
}

// @ismany "$.index[*][?(@.inner.name == 'Foo' && @.kind == 'import')].inner.id" $foo_fn $foo_struct
// @ismany "$.index[*][?(@.inner.name == 'Bar' && @.kind == 'import')].inner.id" $foo_fn $foo_struct

// @count "$.index[*][?(@.inner.name == 'Foo' && @.kind == 'import')]" 2
pub use nested::Foo;
// @count "$.index[*][?(@.inner.name == 'Bar' && @.kind == 'import')]" 2
pub use Foo as Bar;
