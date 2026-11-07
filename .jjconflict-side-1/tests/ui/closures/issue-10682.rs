//@ run-pass
// Regression test for issue #10682
// Nested `proc` usage can't use outer owned data


fn work(_: Box<isize>) {}
fn foo<F:FnOnce()>(_: F) {}

pub fn main() {
  let a = Box::new(1);
  foo(move|| { foo(move|| { work(a) }) })
}
