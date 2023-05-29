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

// @ismany "$.index[*].inner[?(@.import.name == 'Foo')].import.id" $foo_fn $foo_struct
// @ismany "$.index[*].inner[?(@.import.name == 'Bar')].import.id" $foo_fn $foo_struct

// @count "$.index[*].inner[?(@.import.name == 'Foo')]" 2
pub use nested::Foo;
// @count "$.index[*].inner[?(@.import.name == 'Bar')]" 2
pub use Foo as Bar;
