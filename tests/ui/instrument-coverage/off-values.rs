//@ check-pass
//@ revisions: n no off _false zero
//@ [n] compile-flags: -Cinstrument-coverage=n
//@ [no] compile-flags: -Cinstrument-coverage=no
//@ [off] compile-flags: -Cinstrument-coverage=off
//@ [_false] compile-flags: -Cinstrument-coverage=false
//@ [zero] compile-flags: -Cinstrument-coverage=0

fn main() {}
