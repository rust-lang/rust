//! Test case for [#78160].
//!
//! A SomeTrait that is implemented for `&mut T` should not be marked as
//! "notable" for return values that are `&T`.
//!
//! [#78160]: https://github.com/rust-lang/rust/issues/78160

#![feature(rustdoc_internals)]

#[doc(primitive = "reference")]
/// Some useless docs, wouhou!
///
/// We need to put this in here, because notable traits
/// that are implemented on foreign types don't show up.
mod reference {}

// @has doc_notable_trait_mut_t_is_not_ref_t/fn.fn_no_matches.html
// @!has - '//code[@class="content"]' "impl<'_, I> Iterator for &'_ mut I"
pub fn fn_no_matches<'a, T: Iterator + 'a>() -> &'a T {
    loop {}
}
