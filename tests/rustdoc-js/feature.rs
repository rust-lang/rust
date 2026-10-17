// This test ensures that the "features" passed through command line `--feature-documentation` are
// also included into the search index (and results).

//@ compile-flags: -Zunstable-options --feature-documentation x=tadam

#![crate_name = "foo"]
