// Regression test for remapped paths in rustdoc errors
// <https://github.com/rust-lang/rust/issues/69264>.

//@ compile-flags:-Z unstable-options --remap-path-prefix={{src-base}}=remapped_path
//@ rustc-env:RUST_BACKTRACE=0

#![deny(rustdoc::invalid_html_tags)]

/// </script>
pub struct Bar;

//~? ERROR unopened HTML tag `script`
