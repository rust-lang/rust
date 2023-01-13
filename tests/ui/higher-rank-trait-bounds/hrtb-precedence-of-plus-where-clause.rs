// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

// Test that `F : Fn(isize) -> isize + Send` is interpreted as two
// distinct bounds on `F`.

fn foo1<F>(f: F)
    where F : FnOnce(isize) -> isize + Send
{
    bar(f);
}

fn foo2<F>(f: F)
    where F : FnOnce(isize) -> isize + Send
{
    baz(f);
}

fn bar<F:Send>(f: F) { }

fn baz<F:FnOnce(isize) -> isize>(f: F) { }

fn main() {}
