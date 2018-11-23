// edition:2018
// compile-flags:--extern issue_56125
// aux-build:issue-56125.rs

use issue_56125::last_segment::*;
//~^ ERROR `issue_56125` is ambiguous
//~| ERROR unresolved import `issue_56125::last_segment`
use issue_56125::non_last_segment::non_last_segment::*;
//~^ ERROR `issue_56125` is ambiguous
//~| ERROR failed to resolve: could not find `non_last_segment` in `issue_56125`

fn main() {}
