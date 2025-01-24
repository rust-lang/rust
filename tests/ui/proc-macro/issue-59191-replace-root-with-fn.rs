// Test that using a macro to replace the entire crate tree with a non-'mod' item errors out nicely.
// `issue_59191::no_main` replaces whatever's passed in with `fn main() {}`.

//@ edition:2018
//@ proc-macro: issue-59191.rs
//@ error-pattern: requires `sized` lang_item

#![feature(custom_inner_attributes)]
#![issue_59191::no_main]
#![issue_59191::no_main]
