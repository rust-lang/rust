// Test view items inside non-file-level mods

// This is a regression test for an issue where we were failing to
// pretty-print such view items. If that happens again, this should
// begin failing.

mod m {
    use std;
    import vec;
    fn f() -> [int] { vec::init_elt(0, 1u) }
}

fn main() { let x = m::f(); }
