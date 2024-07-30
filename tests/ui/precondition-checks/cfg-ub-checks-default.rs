//@ run-pass
//@ revisions: yes no
//@ [yes] compile-flags: -Cdebug-assertions=yes
//@ [no] compile-flags: -Cdebug-assertions=no

#![feature(cfg_ub_checks)]

fn main() {
    assert_eq!(cfg!(ub_checks), cfg!(debug_assertions));
}
