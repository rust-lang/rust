// compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

// @has 'src/foo/jump-to-def-doc-links.rs.html'

// @has - '//a[@href="../../foo/struct.Bar.html"]' 'Bar'
// @has - '//a[@href="../../foo/struct.Foo.html"]' 'Foo'
pub struct Bar; pub struct Foo;

// @has - '//a[@href="../../foo/enum.Enum.html"]' 'Enum'
pub enum Enum {
    Variant1(String),
    Variant2(u8),
}

// @has - '//a[@href="../../foo/struct.Struct.html"]' 'Struct'
pub struct Struct {
    pub a: u8,
    b: Foo,
}

impl Struct {
    pub fn foo() {}
    pub fn foo2(&self) {}
    fn bar() {}
    fn bar(&self) {}
}

// @has - '//a[@href="../../foo/trait.Trait.html"]' 'Trait'
pub trait Trait {
    fn foo();
}

impl Trait for Struct {
    fn foo() {}
}

// @has - '//a[@href="../../foo/union.Union.html"]' 'Union'
pub union Union {
    pub a: u16,
    pub f: u32,
}

// @has - '//a[@href="../../foo/fn.bar.html"]' 'bar'
pub fn bar(b: Bar) {
     let x = Foo;
}

// @has - '//a[@href="../../foo/bar/index.html"]' 'bar'
pub mod bar {}
