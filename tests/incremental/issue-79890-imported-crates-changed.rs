// aux-build:issue-79890.rs
// revisions:rpass1 rpass2 rpass3
// compile-flags:--extern issue_79890 --test
// edition:2018

// Tests that we don't ICE when the set of imported crates changes
#[cfg(rpass2)] use issue_79890::MyTrait;
