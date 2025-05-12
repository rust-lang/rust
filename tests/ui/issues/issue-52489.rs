//@ edition:2018
//@ aux-build:issue-52489.rs
//@ compile-flags:--extern issue_52489

use issue_52489;
//~^ ERROR use of unstable library feature `issue_52489_unstable`

fn main() {}
