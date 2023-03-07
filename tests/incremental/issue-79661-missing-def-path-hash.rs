// aux-build:issue-79661.rs
// revisions: rpass1 rpass2 rpass3

// Regression test for issue #79661
// We were failing to copy over a DefPathHash->DefId mapping
// from the old incremental cache to the new incremental cache
// when we ended up forcing a query. As a result, a subsequent
// unchanged incremental run would crash due to the missing mapping

extern crate issue_79661;
use issue_79661::Wrapper;

pub struct Outer(Wrapper);
fn main() {}
