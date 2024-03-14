//@ compile-flags: --extern aux_issue_121915 --edition 2015
//@ aux-build: aux-issue-121915.rs

extern crate aux_issue_121915;

#[deny(unused_imports)]
fn main() {
    use aux_issue_121915;
    //~^ ERROR redundant import
    aux_issue_121915::item();
}
