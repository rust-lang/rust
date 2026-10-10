// This is a regression test for issue #89119: an issue in intercrate mode caching.
//
// It requires multiple crates, of course, but the bug is triggered by the code in the dependency,
// not the main crate. This is why this file is empty.
//
// The auxiliary crate used in the test contains the code minimized from `zvariant-2.8.0`.

//@ check-pass
//@ aux-build: issue_89119_intercrate_caching.rs

fn main() {}
