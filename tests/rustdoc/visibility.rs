//@ compile-flags: --document-private-items

#![crate_name = "foo"]
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

//@ !has 'foo/index.html' '//a[@href="struct.FooPublic.html"]/..' 'FooPublic 🔒'
//@ has 'foo/struct.FooPublic.html' '//pre' 'pub struct FooPublic'
pub struct FooPublic;
//@ has 'foo/index.html' '//a[@href="struct.FooJustCrate.html"]/..' 'FooJustCrate 🔒'
//@ has 'foo/struct.FooJustCrate.html' '//pre' 'pub(crate) struct FooJustCrate'
pub(crate) struct FooJustCrate;
//@ has 'foo/index.html' '//a[@href="struct.FooPubCrate.html"]/..' 'FooPubCrate 🔒'
//@ has 'foo/struct.FooPubCrate.html' '//pre' 'pub(crate) struct FooPubCrate'
pub(crate) struct FooPubCrate;
//@ has 'foo/index.html' '//a[@href="struct.FooSelf.html"]/..' 'FooSelf 🔒'
//@ has 'foo/struct.FooSelf.html' '//pre' 'pub(crate) struct FooSelf'
pub(self) struct FooSelf;
//@ has 'foo/index.html' '//a[@href="struct.FooInSelf.html"]/..' 'FooInSelf 🔒'
//@ has 'foo/struct.FooInSelf.html' '//pre' 'pub(crate) struct FooInSelf'
pub(in self) struct FooInSelf;
//@ has 'foo/index.html' '//a[@href="struct.FooPriv.html"]/..' 'FooPriv 🔒'
//@ has 'foo/struct.FooPriv.html' '//pre' 'pub(crate) struct FooPriv'
struct FooPriv;

//@ !has 'foo/index.html' '//a[@href="pub_mod/index.html"]/..' 'pub_mod 🔒'
pub mod pub_mod {}

//@ has 'foo/index.html' '//a[@href="pub_crate_mod/index.html"]/..' 'pub_crate_mod 🔒'
pub(crate) mod pub_crate_mod {}

//@ has 'foo/index.html' '//a[@href="a/index.html"]/..' 'a 🔒'
mod a {
    //@ has 'foo/a/index.html' '//a[@href="struct.FooASuper.html"]/..' 'FooASuper 🔒'
    //@ has 'foo/a/struct.FooASuper.html' '//pre' 'pub(crate) struct FooASuper'
    pub(super) struct FooASuper;
    //@ has 'foo/a/index.html' '//a[@href="struct.FooAInSuper.html"]/..' 'FooAInSuper 🔒'
    //@ has 'foo/a/struct.FooAInSuper.html' '//pre' 'pub(crate) struct FooAInSuper'
    pub(in super) struct FooAInSuper;
    //@ has 'foo/a/index.html' '//a[@href="struct.FooAInA.html"]/..' 'FooAInA 🔒'
    //@ has 'foo/a/struct.FooAInA.html' '//pre' 'struct FooAInA'
    //@ !has 'foo/a/struct.FooAInA.html' '//pre' 'pub'
    pub(in a) struct FooAInA;
    //@ has 'foo/a/index.html' '//a[@href="struct.FooAPriv.html"]/..' 'FooAPriv 🔒'
    //@ has 'foo/a/struct.FooAPriv.html' '//pre' 'struct FooAPriv'
    //@ !has 'foo/a/struct.FooAPriv.html' '//pre' 'pub'
    struct FooAPriv;

    //@ has 'foo/a/index.html' '//a[@href="b/index.html"]/..' 'b 🔒'
    mod b {
        //@ has 'foo/a/b/index.html' '//a[@href="struct.FooBSuper.html"]/..' 'FooBSuper 🔒'
        //@ has 'foo/a/b/struct.FooBSuper.html' '//pre' 'pub(super) struct FooBSuper'
        pub(super) struct FooBSuper;
        //@ has 'foo/a/b/index.html' '//a[@href="struct.FooBInSuperSuper.html"]/..' 'FooBInSuperSuper 🔒'
        //@ has 'foo/a/b/struct.FooBInSuperSuper.html' '//pre' 'pub(crate) struct FooBInSuperSuper'
        pub(in super::super) struct FooBInSuperSuper;
        //@ has 'foo/a/b/index.html' '//a[@href="struct.FooBInAB.html"]/..' 'FooBInAB 🔒'
        //@ has 'foo/a/b/struct.FooBInAB.html' '//pre' 'struct FooBInAB'
        //@ !has 'foo/a/b/struct.FooBInAB.html' '//pre' 'pub'
        pub(in a::b) struct FooBInAB;
        //@ has 'foo/a/b/index.html' '//a[@href="struct.FooBPriv.html"]/..' 'FooBPriv 🔒'
        //@ has 'foo/a/b/struct.FooBPriv.html' '//pre' 'struct FooBPriv'
        //@ !has 'foo/a/b/struct.FooBPriv.html' '//pre' 'pub'
        struct FooBPriv;

        //@ !has 'foo/a/b/index.html' '//a[@href="struct.FooBPub.html"]/..' 'FooBPub 🔒'
        //@ has 'foo/a/b/struct.FooBPub.html' '//pre' 'pub struct FooBPub'
        pub struct FooBPub;
    }
}

//@ has 'foo/trait.PubTrait.html' '//pre' 'pub trait PubTrait'
//
//@ has 'foo/trait.PubTrait.html' '//pre' 'type Type;'
//@ !has 'foo/trait.PubTrait.html' '//pre' 'pub type Type;'
//
//@ has 'foo/trait.PubTrait.html' '//pre' 'const CONST: usize;'
//@ !has 'foo/trait.PubTrait.html' '//pre' 'pub const CONST: usize;'
//
//@ has 'foo/trait.PubTrait.html' '//pre' 'fn function();'
//@ !has 'foo/trait.PubTrait.html' '//pre' 'pub fn function();'
//
//@ !has 'foo/index.html' '//a[@href="trait.PubTrait.html"]/..' 'PubTrait 🔒'

pub trait PubTrait {
    type Type;
    const CONST: usize;
    fn function();
}

//@ has 'foo/index.html' '//a[@href="trait.PrivTrait.html"]/..' 'PrivTrait 🔒'
trait PrivTrait {}

//@ has 'foo/struct.FooPublic.html' '//h4[@class="code-header"]' 'type Type'
//@ !has 'foo/struct.FooPublic.html' '//h4[@class="code-header"]' 'pub type Type'
//
//@ has 'foo/struct.FooPublic.html' '//h4[@class="code-header"]' 'const CONST: usize'
//@ !has 'foo/struct.FooPublic.html' '//h4[@class="code-header"]' 'pub const CONST: usize'
//
//@ has 'foo/struct.FooPublic.html' '//h4[@class="code-header"]' 'fn function()'
//@ !has 'foo/struct.FooPublic.html' '//h4[@class="code-header"]' 'pub fn function()'

impl PubTrait for FooPublic {
    type Type = usize;
    const CONST: usize = 0;
    fn function() {}
}

pub struct Assoc;

//@ has foo/struct.Assoc.html
impl Assoc {
    //@ has - '//*[@id="associatedtype.TypePub"]' 'pub type TypePub'
    pub type TypePub = usize;

    //@ has - '//*[@id="associatedtype.TypePriv"]' 'pub(crate) type TypePriv'
    type TypePriv = usize;

    //@ has - '//*[@id="associatedconstant.CONST_PUB"]' 'pub const CONST_PUB'
    pub const CONST_PUB: usize = 0;

    //@ has - '//*[@id="associatedconstant.CONST_PRIV"]' 'pub(crate) const CONST_PRIV'
    const CONST_PRIV: usize = 0;

    //@ has - '//*[@id="method.function_pub"]' 'pub fn function_pub()'
    pub fn function_pub() {}

    //@ has - '//*[@id="method.function_priv"]' 'pub(crate) fn function_priv()'
    fn function_priv() {}
}
