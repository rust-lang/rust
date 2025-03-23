//@ edition: 2021
//@ aux-crate:issue_127011_a=issue-127011-a.rs
//@ aux-crate:issue_127011_b=issue-127011-b.rs

use issue_127011_b::Bar;

fn main() {
    Bar::foo();
    //~^ ERROR: no function or associated item named `foo` found for struct `Bar` in the current scope [E0599]
}
