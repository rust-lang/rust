#![feature(doc_notable_trait)]

#[doc(notable_trait)]
pub trait SomeTrait {}

pub struct SomeStruct;
pub struct OtherStruct;
impl SomeTrait for &[SomeStruct] {}

// @has doc_notable_trait_slice/fn.bare_fn_matches.html
// @has - '//code[@class="content"]' 'impl SomeTrait for &[SomeStruct]'
pub fn bare_fn_matches() -> &'static [SomeStruct] {
    &[]
}

// @has doc_notable_trait_slice/fn.bare_fn_no_matches.html
// @!has - '//code[@class="content"]' 'impl SomeTrait for &[SomeStruct]'
pub fn bare_fn_no_matches() -> &'static [OtherStruct] {
    &[]
}
