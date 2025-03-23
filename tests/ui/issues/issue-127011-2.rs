//@ edition: 2021
//@ aux-crate:issue_127011_a_2=issue-127011-a-2.rs
//@ aux-crate:issue_127011_b_2=issue-127011-b-2.rs

use issue_127011_b_2::Bar;

fn main() {
    Bar::foo();
    //~^ ERROR: no function or associated item named `foo` found for struct `Bar` in the current scope [E0599]
}
