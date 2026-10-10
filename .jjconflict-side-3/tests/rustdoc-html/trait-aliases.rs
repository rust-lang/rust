// Basic testing for trait aliases.
#![feature(trait_alias)]
#![crate_name = "it"]

// Check the "local case" (HIR cleaning) //

//@ has it/all.html '//a[@href="traitalias.Alias0.html"]' 'Alias0'
//@ has it/index.html '//h2[@id="trait-aliases"]' 'Trait Aliases'
//@ has it/index.html '//a[@class="traitalias"]' 'Alias0'
//@ has it/traitalias.Alias0.html
//@ has - '//*[@class="rust item-decl"]//code' 'trait Alias0 = Copy + Iterator<Item = u8>;'
pub trait Alias0 = Copy + Iterator<Item = u8>;

//@ has it/traitalias.Alias1.html
//@ has - '//pre[@class="rust item-decl"]' \
//        "trait Alias1<'a, T: 'a + Clone, const N: usize> = From<[&'a T; N]>;"
pub trait Alias1<'a, T: 'a + Clone, const N: usize> = From<[&'a T; N]>;

//@ has it/traitalias.Alias2.html
//@ has - '//pre[@class="rust item-decl"]' \
//        'trait Alias2<T> = where T: From<String>, String: Into<T>;'
pub trait Alias2<T> = where T: From<String>, String: Into<T>;

//@ has it/traitalias.Alias3.html
//@ has - '//pre[@class="rust item-decl"]' 'trait Alias3 = ;'
pub trait Alias3 =;

//@ has it/traitalias.Alias4.html
//@ has - '//pre[@class="rust item-decl"]' 'trait Alias4 = ;'
pub trait Alias4 = where;

//@ has it/fn.usage0.html
//@ has - '//pre[@class="rust item-decl"]' "pub fn usage0(_: impl Alias0)"
//@ has - '//a[@href="traitalias.Alias0.html"]' 'Alias0'
pub fn usage0(_: impl Alias0) {}

// FIXME: One can only "disambiguate" intra-doc links to trait aliases with `type@` but not with
// `trait@` (fails to resolve) or `traitalias@` (doesn't exist). We should make at least one of
// the latter two work, right?

//@ has it/link0/index.html
//@ has - '//a/@href' 'traitalias.Alias0.html'
//@ has - '//a/@href' 'traitalias.Alias1.html'
/// [Alias0], [type@Alias1]
pub mod link0 {}

// Check the "extern case" (middle cleaning) //

//@ aux-build: ext-trait-aliases.rs
extern crate ext_trait_aliases as ext;

//@ has it/traitalias.ExtAlias0.html
//@ has - '//pre[@class="rust item-decl"]' 'trait ExtAlias0 = Copy + Iterator<Item = u8>;'
pub use ext::ExtAlias0;

//@ has it/traitalias.ExtAlias1.html
//@ has - '//pre[@class="rust item-decl"]' \
//        "trait ExtAlias1<'a, T, const N: usize> = From<[&'a T; N]> where T: 'a + Clone;"
pub use ext::ExtAlias1;

//@ has it/traitalias.ExtAlias2.html
//@ has - '//pre[@class="rust item-decl"]' \
//        'trait ExtAlias2<T> = where T: From<String>, String: Into<T>;'
pub use ext::ExtAlias2;

//@ has it/traitalias.ExtAlias3.html
//@ has - '//pre[@class="rust item-decl"]' 'trait ExtAlias3 = Sized;'
pub use ext::ExtAlias3;

// NOTE: Middle cleaning can't discern `= Sized` and `= where Self: Sized` and that's okay.
//@ has it/traitalias.ExtAlias4.html
//@ has - '//pre[@class="rust item-decl"]' 'trait ExtAlias4 = Sized;'
pub use ext::ExtAlias4;

//@ has it/traitalias.ExtAlias5.html
//@ has - '//pre[@class="rust item-decl"]' 'trait ExtAlias5 = ;'
pub use ext::ExtAlias5;

//@ has it/fn.usage1.html
//@ has - '//pre[@class="rust item-decl"]' "pub fn usage1(_: impl ExtAlias0)"
//@ has - '//a[@href="traitalias.ExtAlias0.html"]' 'ExtAlias0'
pub fn usage1(_: impl ExtAlias0) {}
