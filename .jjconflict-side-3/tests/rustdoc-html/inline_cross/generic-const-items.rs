#![crate_name = "user"]

//@ aux-crate:generic_const_items=generic-const-items.rs
//@ edition:2021

//@ has 'user/constant.K.html'
//@ has - '//*[@class="rust item-decl"]//code' \
// "pub const K<'a, T: 'a + Copy, const N: usize>: Option<[T; N]> \
// where \
//     String: From<T>;"
pub use generic_const_items::K;

//@ has user/trait.Trait.html
//@ has - '//*[@id="associatedconstant.C"]' \
// "const C<'a>: &'a T \
// where \
//     T: 'a + Eq"
pub use generic_const_items::Trait;

//@ has user/struct.Implementor.html
//@ has - '//h3[@class="code-header"]' 'impl Trait<str> for Implementor'
//@ has - '//*[@id="associatedconstant.C"]' \
// "const C<'a>: &'a str = \"C\" \
// where \
//     str: 'a"
pub use generic_const_items::Implementor;
