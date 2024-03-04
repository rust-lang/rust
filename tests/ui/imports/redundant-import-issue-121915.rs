//@ revisions: edition2018 edition2021
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021
//@ compile-flags: --extern aux_issue_121915
//@ aux-build: aux-issue-121915.rs

#[deny(unused_imports)]
fn main() {
    use aux_issue_121915;
    //~^ ERROR the item `aux_issue_121915` is already exists in the extern prelude
    aux_issue_121915::item();
}
