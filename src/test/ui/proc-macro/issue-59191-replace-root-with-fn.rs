// edition:2018
// aux-crate:issue_59191=issue-59191.rs
// Test that using a macro to replace the entire crate tree with a non-'mod' item errors out nicely.
// `issue_59191::no_main` replaces whatever's passed in with `fn main() {}`.
#![feature(custom_inner_attributes)]
//~^ ERROR `main` function not found in crate `issue_59191_replace_root_with_fn` [E0601]
#![issue_59191::no_main]
//~^ ERROR expected crate top-level item to be a module after macro expansion, found a function
