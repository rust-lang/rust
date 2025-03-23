//@ edition: 2021
//@ aux-crate:issue_127011_a_1=issue-127011-a-1.rs
//@ aux-crate:issue_127011_b_1=issue-127011-b-1.rs

use issue_127011_b_1;

fn main() {
    foo();
    //~^ ERROR: cannot find function `foo` in this scope [E0425]
}
