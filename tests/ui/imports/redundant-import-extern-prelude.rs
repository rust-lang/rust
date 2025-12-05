// Check that we detect imports that are redundant due to the extern prelude
// and that we emit a reasonable diagnostic.
// issue: rust-lang/rust#121915
//~^^^ NOTE the item `aux_issue_121915` is already defined by the extern prelude

// See also the discussion in <https://github.com/rust-lang/rust/pull/122954>.

//@ compile-flags: --extern aux_issue_121915
//@ edition: 2018
//@ aux-build: aux-issue-121915.rs

#[deny(redundant_imports)]
//~^ NOTE the lint level is defined here
fn main() {
    use aux_issue_121915;
    //~^ ERROR the item `aux_issue_121915` is imported redundantly
    aux_issue_121915::item();
}
