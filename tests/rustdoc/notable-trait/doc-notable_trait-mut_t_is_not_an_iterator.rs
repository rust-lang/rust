//! Test case for [#80737].
//!
//! A SomeTrait that is implemented for `&mut T where T: SomeTrait`
//! should not be marked as "notable" for return values that do not
//! have bounds on the trait itself.
//!
//! [#80737]: https://github.com/rust-lang/rust/issues/80737

#![feature(rustc_attrs)]
#![no_std]

#[rustc_doc_primitive = "reference"]
/// Some useless docs, wouhou!
///
/// We need to put this in here, because notable traits
/// that are implemented on foreign types don't show up.
mod reference {}

//@ has doc_notable_trait_mut_t_is_not_an_iterator/fn.fn_no_matches.html
//@ !has - '//code[@class="content"]' 'Iterator'
pub fn fn_no_matches<'a, T: 'a>() -> &'a mut T {
    panic!()
}
