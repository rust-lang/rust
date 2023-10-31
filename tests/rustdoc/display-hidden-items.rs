// Test to ensure that the `--document-hidden-items` option is working as expected.
// compile-flags: -Z unstable-options --document-hidden-items
// ignore-tidy-linelength

#![crate_name = "foo"]

// @has 'foo/index.html'
// @has - '//*[@id="reexport.hidden_reexport"]/code' 'pub use hidden::inside_hidden as hidden_reexport;'
#[doc(hidden)]
pub use hidden::inside_hidden as hidden_reexport;

// @has - '//*[@class="item-name"]/a[@class="trait"]' 'TraitHidden'
// @has 'foo/trait.TraitHidden.html'
#[doc(hidden)]
pub trait TraitHidden {}

// @has 'foo/index.html' '//*[@class="item-name"]/a[@class="trait"]' 'Trait'
pub trait Trait {
    // @has 'foo/trait.Trait.html'
    // @has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]' 'const BAR: u32 = 0u32'
    #[doc(hidden)]
    const BAR: u32 = 0;

    // @has - '//*[@id="method.foo"]/*[@class="code-header"]' 'fn foo()'
    #[doc(hidden)]
    fn foo() {}
}

// @has 'foo/index.html' '//*[@class="item-name"]/a[@class="struct"]' 'Struct'
// @has 'foo/struct.Struct.html'
pub struct Struct {
    // @has - '//*[@id="structfield.a"]/code' 'a: u32'
    #[doc(hidden)]
    pub a: u32,
}

impl Struct {
    // @has - '//*[@id="method.new"]/*[@class="code-header"]' 'pub fn new() -> Self'
    #[doc(hidden)]
    pub fn new() -> Self { Self { a: 0 } }
}

impl Trait for Struct {
    // @has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]' 'const BAR: u32 = 0u32'
    // @has - '//*[@id="method.foo"]/*[@class="code-header"]' 'fn foo()'
}
// @has - '//*[@id="impl-TraitHidden-for-Struct"]/*[@class="code-header"]' 'impl TraitHidden for Struct'
impl TraitHidden for Struct {}

// @has 'foo/index.html' '//*[@class="item-name"]/a[@class="enum"]' 'HiddenEnum'
// @has 'foo/enum.HiddenEnum.html'
#[doc(hidden)]
pub enum HiddenEnum {
    A,
}

// @has 'foo/index.html' '//*[@class="item-name"]/a[@class="enum"]' 'Enum'
pub enum Enum {
    // @has 'foo/enum.Enum.html' '//*[@id="variant.A"]/*[@class="code-header"]' 'A'
    #[doc(hidden)]
    A,
}

// @has 'foo/index.html' '//*[@class="item-name"]/a[@class="mod"]' 'hidden'
#[doc(hidden)]
pub mod hidden {
    // @has 'foo/hidden/index.html'
    // @has - '//*[@class="item-name"]/a[@class="fn"]' 'inside_hidden'
    // @has 'foo/hidden/fn.inside_hidden.html'
    pub fn inside_hidden() {}
}
