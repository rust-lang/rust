#![no_core]
#![feature(no_core)]
#![feature(staged_api)]
#![doc(test(no_crate_inject))]

/// Returns whether the given expression matches (any of) the given pattern(s).
///
/// # Examples
///
/// ```
/// #![feature(matches_macro)]
/// use std::macros::matches;
///
/// let foo = 'f';
/// assert!(matches!(foo, 'A'..='Z' | 'a'..='z'));
///
/// let bar = Some(4);
/// assert!(matches!(bar, Some(x) if x > 2));
/// ```
#[macro_export]
#[unstable(feature = "matches_macro", issue = "0")]
macro_rules! matches {
    ($expression:expr, $( $pattern:pat )|+ $( if $guard: expr )?) => {
        match $expression {
            $( $pattern )|+ $( if $guard )? => true,
            _ => false
        }
    }
}
