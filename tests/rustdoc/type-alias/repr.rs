// This test ensures that the `repr` attribute is displayed in type aliases.
//
// Regression test for <https://github.com/rust-lang/rust/issues/140739>.

#![crate_name = "foo"]

/// bla
#[repr(C)]
pub struct Foo1;

//@ has 'foo/type.Bar1.html'
//@ has - '//*[@class="rust item-decl"]/code' '#[repr(C)]pub struct Bar1;'
// Ensures that we see the doc comment of the type alias and not of the aliased type.
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'bar'
/// bar
pub type Bar1 = Foo1;

/// bla
#[repr(C)]
pub union Foo2 {
    pub a: u8,
}

//@ has 'foo/type.Bar2.html'
//@ matches - '//*[@class="rust item-decl"]' '#\[repr\(C\)\]\npub union Bar2 \{*'
// Ensures that we see the doc comment of the type alias and not of the aliased type.
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'bar'
/// bar
pub type Bar2 = Foo2;

/// bla
#[repr(C)]
pub enum Foo3 {
    A,
}

//@ has 'foo/type.Bar3.html'
//@ matches - '//*[@class="rust item-decl"]' '#\[repr\(C\)\]pub enum Bar3 \{*'
// Ensures that we see the doc comment of the type alias and not of the aliased type.
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'bar'
/// bar
pub type Bar3 = Foo3;
