// edition: 2021
// compile-flags: --extern issue_85992_extern_1 --extern issue_85992_extern_2
// aux-build: issue-85992-extern-1.rs
// aux-build: issue-85992-extern-2.rs

issue_85992_extern_1::m!();

use crate::issue_85992_extern_2;
//~^ ERROR unresolved import `crate::issue_85992_extern_2`

fn main() {}
