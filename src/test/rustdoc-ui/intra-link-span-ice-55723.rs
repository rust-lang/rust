// ignore-tidy-end-whitespace

#![deny(intra_doc_link_resolution_failure)]

// An error in calculating spans while reporting intra-doc link resolution errors caused rustdoc to
// attempt to slice in the middle of a multibyte character.

/// ## For example:
///
/// （arr[i]）
//~^ ERROR `[i]` cannot be resolved, ignoring it...
pub fn test_ice() {
    unimplemented!();
}
