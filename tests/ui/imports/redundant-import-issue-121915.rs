//@ compile-flags: --extern aux_issue_121915 --edition 2018
//@ aux-build: aux-issue-121915.rs

#[deny(unused_imports)]
fn main() {
    use aux_issue_121915;
    //~^ ERROR redundant import
    aux_issue_121915::item();
}
