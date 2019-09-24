// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Regression test for #24085. Errors were occurring in region
// inference due to the requirement that `'a:b'`, which was getting
// incorrectly codegened in connection with the closure below.

#[derive(Copy,Clone)]
struct Path<'a:'b, 'b> {
    x: &'a i32,
    tail: Option<&'b Path<'a, 'b>>
}

#[allow(dead_code, unconditional_recursion)]
fn foo<'a,'b,F>(p: Path<'a, 'b>, mut f: F)
                where F: for<'c> FnMut(Path<'a, 'c>) {
    foo(p, |x| f(x))
}

fn main() { }
