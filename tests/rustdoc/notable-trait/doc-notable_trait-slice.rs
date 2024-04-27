#![feature(doc_notable_trait)]

#[doc(notable_trait)]
pub trait SomeTrait {}

pub struct SomeStruct;
pub struct OtherStruct;
impl SomeTrait for &[SomeStruct] {}

// @has doc_notable_trait_slice/fn.bare_fn_matches.html
// @snapshot bare_fn_matches - '//script[@id="notable-traits-data"]'
pub fn bare_fn_matches() -> &'static [SomeStruct] {
    &[]
}

// @has doc_notable_trait_slice/fn.bare_fn_no_matches.html
// @count - '//script[@id="notable-traits-data"]' 0
pub fn bare_fn_no_matches() -> &'static [OtherStruct] {
    &[]
}

// @has doc_notable_trait_slice/fn.bare_fn_mut_no_matches.html
// @count - '//script[@id="notable-traits-data"]' 0
pub fn bare_fn_mut_no_matches() -> &'static mut [SomeStruct] {
    &mut []
}
