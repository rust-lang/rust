// Test to ensure that the `--document-hidden-items` option is working as expected.
//@ compile-flags: -Z unstable-options --document-hidden-items
// ignore-tidy-linelength

#![crate_name = "foo"]

//@ has 'foo/index.html'

//@ matches - '//dt[code]' 'pub extern crate .*hidden_core;.*👻'
//@ has - '//dt/code' 'pub extern crate core as hidden_core;'
#[doc(hidden)]
pub extern crate core as hidden_core;

//@ has - '//*[@id="reexport.hidden_reexport"]/span[@title="Hidden item"]' '👻'
//@ has - '//*[@id="reexport.hidden_reexport"]/code' 'pub use hidden::inside_hidden as hidden_reexport;'
#[doc(hidden)]
pub use hidden::inside_hidden as hidden_reexport;

//@ has - '//dt/a[@class="trait"]' 'TraitHidden'
//@ has 'foo/trait.TraitHidden.html'
//@ has 'foo/trait.TraitHidden.html' '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[doc(hidden)]'
//@ has 'foo/trait.TraitHidden.html' '//*[@class="rust item-decl"]/code' 'pub trait TraitHidden'
#[doc(hidden)]
pub trait TraitHidden {}

//@ has 'foo/index.html' '//dt/a[@class="trait"]' 'Trait'
pub trait Trait {
    //@ has 'foo/trait.Trait.html'
    //@ has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    const BAR: u32 = 0;

    //@ has - '//*[@id="method.foo"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    //@ has - '//*[@id="method.foo"]/*[@class="code-header"]' 'fn foo()'
    #[doc(hidden)]
    fn foo() {}
}

//@ has 'foo/index.html' '//dt/a[@class="struct"]' 'Struct'
//@ has 'foo/struct.Struct.html'
pub struct Struct {
    //@ has - '//*[@id="structfield.a"]/code' 'a: u32'
    #[doc(hidden)]
    pub a: u32,
}

impl Struct {
    //@ has - '//*[@id="method.new"]/*[@class="code-header"]' 'pub fn new() -> Self'
    #[doc(hidden)]
    pub fn new() -> Self { Self { a: 0 } }
}

impl Trait for Struct {
    //@ has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    //@ has - '//*[@id="method.foo"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
}
//@ has - '//*[@id="impl-TraitHidden-for-Struct"]/*[@class="code-header"]' 'impl TraitHidden for Struct'
impl TraitHidden for Struct {}

//@ has 'foo/index.html' '//dt/a[@class="enum"]' 'HiddenEnum'
//@ has 'foo/enum.HiddenEnum.html'
//@ has 'foo/enum.HiddenEnum.html' '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[doc(hidden)]'
//@ has 'foo/enum.HiddenEnum.html' '//*[@class="rust item-decl"]/code' 'pub enum HiddenEnum'
#[doc(hidden)]
pub enum HiddenEnum {
    A,
}

//@ has 'foo/index.html' '//dt/a[@class="enum"]' 'Enum'
pub enum Enum {
    //@ has 'foo/enum.Enum.html' '//*[@id="variant.A"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    //@ has 'foo/enum.Enum.html' '//*[@id="variant.A"]/*[@class="code-header"]' 'A'
    #[doc(hidden)]
    A,
}

//@ has 'foo/index.html' '//dt/a[@class="mod"]' 'hidden'
#[doc(hidden)]
pub mod hidden {
    //@ has 'foo/hidden/index.html'
    //@ has - '//dt/a[@class="fn"]' 'inside_hidden'
    //@ has 'foo/hidden/fn.inside_hidden.html'
    pub fn inside_hidden() {}
}
