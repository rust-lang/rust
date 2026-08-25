//@ compile-flags: --extern aux_issue_121915
//@ edition: 2015
//@ aux-build: aux-issue-121915.rs

extern crate aux_issue_121915;

#[deny(redundant_imports)]
fn main() {
    use aux_issue_121915;
    //~^ ERROR the item `aux_issue_121915` is imported redundantly
    aux_issue_121915::item();
}
