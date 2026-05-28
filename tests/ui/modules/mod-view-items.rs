//@ run-pass
// Test view items inside non-file-level mods

// This is a regression test for an issue where we were failing to
// pretty-print such view items. If that happens again, this should
// begin failing.


mod m {
    pub fn f() -> Vec<isize> { Vec::new() }
}

pub fn main() { let _x = m::f(); }
