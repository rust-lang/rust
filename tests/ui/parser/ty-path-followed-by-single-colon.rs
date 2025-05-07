// Paths in type contexts may be followed by single colons.
// This means we can't generally assume that the user typo'ed a double colon.
// issue: <https://github.com/rust-lang/rust/issues/140227>
//@ check-pass
#![crate_type = "lib"]
#![expect(non_camel_case_types)]

#[rustfmt::skip]
mod garden {

    fn f<path>() where path:to::somewhere {} // OK!

    fn g(_: impl Take<path:to::somewhere>) {} // OK!

    #[cfg(any())] fn h() where a::path:to::nowhere {} // OK!

    fn i(_: impl Take<path::<>:to::somewhere>) {} // OK!

    mod to { pub(super) trait somewhere {} }
    trait Take { type path; }

}
