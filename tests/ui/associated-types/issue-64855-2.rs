// This was originally a test for a `ReEmpty` ICE, but became an unintentional test of
// the coinductiveness of WF predicates. That behavior was removed, and thus this is
// also inadvertently a test for the (non-)co-inductiveness of WF predicates.

pub struct Bar<'a>(&'a Self) where Self: ;
//~^ ERROR overflow evaluating the requirement `Bar<'a> well-formed`

fn main() {}
