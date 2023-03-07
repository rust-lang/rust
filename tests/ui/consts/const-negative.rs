// run-pass
// Issue #358
#![allow(non_upper_case_globals)]

static toplevel_mod: isize = -1;

pub fn main() {
    assert_eq!(toplevel_mod, -1);
}
