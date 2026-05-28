//@ check-pass
//@ compile-flags: -Zno-profiler-runtime
//@ revisions: default y yes on true_ all
//@ [default] compile-flags: -Cinstrument-coverage
//@ [y] compile-flags: -Cinstrument-coverage=y
//@ [yes] compile-flags: -Cinstrument-coverage=yes
//@ [on] compile-flags: -Cinstrument-coverage=on
//@ [true_] compile-flags: -Cinstrument-coverage=true
//@ [all] compile-flags: -Cinstrument-coverage=all

fn main() {}
