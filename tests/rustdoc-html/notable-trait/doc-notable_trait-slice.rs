#![feature(doc_notable_trait)]

#[doc(notable_trait)]
pub trait SomeTrait {}

pub struct SomeStruct;
pub struct OtherStruct;
impl SomeTrait for &[SomeStruct] {}

//@ has doc_notable_trait_slice/fn.bare_fn_matches.notable-traits-data.js
//@ snapshot bare_fn_matches -
pub fn bare_fn_matches() -> &'static [SomeStruct] {
    &[]
}

//@ has doc_notable_trait_slice/fn.bare_fn_no_matches.html
//@ !has doc_notable_trait_slice/fn.bare_fn_no_matches.notable-traits-data.js
pub fn bare_fn_no_matches() -> &'static [OtherStruct] {
    &[]
}

//@ has doc_notable_trait_slice/fn.bare_fn_mut_no_matches.html
//@ !has doc_notable_trait_slice/fn.bare_fn_mut_no_matches.notable-traits-data.js
pub fn bare_fn_mut_no_matches() -> &'static mut [SomeStruct] {
    &mut []
}
