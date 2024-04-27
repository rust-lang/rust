// This test ensures that there are 4 imports as expected:
// * 2 for `Foo`
// * 2 for `Bar`

#![crate_name = "foo"]

// @has 'foo/index.html'

pub mod nested {
    /// Foo the struct
    pub struct Foo {}

    #[allow(non_snake_case)]
    /// Foo the function
    pub fn Foo() {}
}

// @count - '//*[@id="main-content"]//code' 'pub use nested::Foo;' 2
// @has - '//*[@id="reexport.Foo"]//a[@href="nested/struct.Foo.html"]' 'Foo'
// @has - '//*[@id="reexport.Foo-1"]//a[@href="nested/fn.Foo.html"]' 'Foo'
pub use nested::Foo;

// @count - '//*[@id="main-content"]//code' 'pub use Foo as Bar;' 2
// @has - '//*[@id="reexport.Bar"]//a[@href="nested/struct.Foo.html"]' 'Foo'
// @has - '//*[@id="reexport.Bar-1"]//a[@href="nested/fn.Foo.html"]' 'Foo'
pub use Foo as Bar;
