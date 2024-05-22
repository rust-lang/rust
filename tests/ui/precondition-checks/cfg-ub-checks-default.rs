//@ run-pass
//@ revisions: YES NO
//@ [YES] compile-flags: -Cdebug-assertions=yes
//@ [NO] compile-flags: -Cdebug-assertions=no

#![feature(cfg_ub_checks)]

fn main() {
    assert_eq!(cfg!(ub_checks), cfg!(debug_assertions));
}
