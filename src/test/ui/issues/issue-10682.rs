// run-pass
// Regression test for issue #10682
// Nested `proc` usage can't use outer owned data

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

fn work(_: Box<isize>) {}
fn foo<F:FnOnce()>(_: F) {}

pub fn main() {
  let a = box 1;
  foo(move|| { foo(move|| { work(a) }) })
}
