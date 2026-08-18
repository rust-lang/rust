#![feature(allow_internal_unstable, edition_redirect, staged_api)]
#![stable(feature = "edition_redirect_stability", since = "1.0.0")]

#[doc(hidden)]
#[unstable(feature = "edition_redirect_old", issue = "none")]
#[macro_export]
macro_rules! old_macro {
    () => { 1 };
}

#[rustc_edition_redirect = "2024"]
#[stable(feature = "edition_redirect_stability", since = "1.0.0")]
pub use old_macro as redirected_macro;

#[stable(feature = "edition_redirect_stability", since = "1.0.0")]
#[allow_internal_unstable(edition_redirect_old)]
#[macro_export]
macro_rules! redirected_macro {
    () => { 2 };
}
