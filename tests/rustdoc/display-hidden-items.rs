// Test to ensure that the `--document-hidden-items` option is working as expected.
//@ compile-flags: -Z unstable-options --document-hidden-items
// ignore-tidy-linelength

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@class="item-name"]/span[@title="Hidden item"]' '👻'

//@ has - '//*[@id="reexport.hidden_reexport"]/code' 'pub use hidden::inside_hidden as hidden_reexport;'
#[doc(hidden)]
pub use hidden::inside_hidden as hidden_reexport;

//@ has - '//*[@class="item-name"]/a[@class="trait"]' 'TraitHidden'
//@ has 'foo/trait.TraitHidden.html' '//*[@class="rust item-decl"]/code' '#[doc(hidden)]'
#[doc(hidden)]
pub trait TraitHidden {}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="trait"]' 'Trait'
pub trait Trait {
    //@ has 'foo/trait.Trait.html'
    //@ has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    const BAR: u32 = 0;

    #[doc(hidden)]
    //@ has 'foo/trait.Trait.html'
    //@ has - '//*[@id="associatedtype.Baz"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    type Baz;

    //@ has - '//*[@id="method.foo"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    fn foo() {}
}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="struct"]' 'Struct'
//@ has 'foo/struct.Struct.html'
pub struct Struct {
    // FIXME: display attrs on fields in the Fields section
    //@ !has - '//*[@id="structfield.a"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    pub a: u32,
}

impl Struct {
    //@ has - '//*[@id="method.new"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    pub fn new() -> Self { Self { a: 0 } }
}

impl Trait for Struct {
    //@ has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    //@ has - '//*[@id="method.foo"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    // NOTE: feature(associated_type_defaults) is unstable so users have to set a type for trait associated items:
    // we don't want to hide it if they don't ask for it explicitely in the impl.
    //@ !has - '//*[@id="associatedtype.Baz"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'

    type Baz = ();
}
//@ has - '//*[@id="impl-TraitHidden-for-Struct"]/*[@class="code-header"]' 'impl TraitHidden for Struct'
impl TraitHidden for Struct {}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="struct"]' 'TupleStruct'
pub struct TupleStruct(
    // FIXME: display attrs on fields in the Fields section
    //@ !has 'foo/struct.TupleStruct.html' '//*[@id="structfield.0"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    ()
);

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="enum"]' 'HiddenEnum'
//@ has 'foo/enum.HiddenEnum.html' '//*[@class="rust item-decl"]/code' '#[doc(hidden)]'
#[doc(hidden)]
pub enum HiddenEnum {
    A,
}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="enum"]' 'Enum'
pub enum Enum {
    //@ has 'foo/enum.Enum.html' '//*[@id="variant.A"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    A,
    //@ has 'foo/enum.Enum.html' '//*[@id="variant.B"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    B(#[doc(hidden)] ())
}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="mod"]' 'hidden'
#[doc(hidden)]
pub mod hidden {
    //@ has 'foo/hidden/index.html'
    //@ has - '//*[@class="item-name"]/a[@class="fn"]' 'inside_hidden'
    //@ has 'foo/hidden/fn.inside_hidden.html'
    pub fn inside_hidden() {}
}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="union"]' 'Union'
//@ has 'foo/union.Union.html' '//*[@class="rust item-decl"]/code' '#[doc(hidden)]'
#[doc(hidden)]
pub union Union {
    // FIXME: display attrs on fields in the Fields section
    //@ !has 'foo/union.Union.html' '//*[@id="structfield.a"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[doc(hidden)]'
    #[doc(hidden)]
    pub a: u8,
}

//@ has 'foo/index.html' '//*[@class="item-name"]/a[@class="macro"]' 'macro_rule'
//@ has 'foo/macro.macro_rule.html' '//*[@class="rust item-decl"]/code' '#[doc(hidden)]'
#[doc(hidden)]
#[macro_export]
macro_rules! macro_rule {
    () => {};
}

