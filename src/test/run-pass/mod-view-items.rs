// Test view items inside non-file-level mods

// This is a regression test for an issue where we were failing to
// pretty-print such view items. If that happens again, this should
// begin failing.

mod m {
    use std;
    import std::vec;
    fn f() -> vec[int] { vec::empty[int]() }
}

fn main() { let x = m::f(); }