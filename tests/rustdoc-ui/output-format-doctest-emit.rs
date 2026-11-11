// Ensure that `--output-format=doctest` is incompatible with the `--emit` flag (for now at least).

//@ revisions: html-static-files dep-info
//@ compile-flags: -Zunstable-options --output-format doctest
//@[html-static-files] compile-flags: --emit html-static-files
//@[dep-info] compile-flags: --emit dep-info

//~? ERROR the `--emit` flag is not supported with `--output-format=doctest`
