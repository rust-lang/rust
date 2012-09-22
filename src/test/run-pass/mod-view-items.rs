// Test view items inside non-file-level mods

// This is a regression test for an issue where we were failing to
// pretty-print such view items. If that happens again, this should
// begin failing.

mod m {
    #[legacy_exports];
    use core::vec;
    fn f() -> ~[int] { vec::from_elem(1u, 0) }
}

fn main() { let x = m::f(); }
