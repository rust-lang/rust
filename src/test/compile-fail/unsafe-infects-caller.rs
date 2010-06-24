// error-pattern: calculated effect is 'unsafe'

native mod foo {
  fn naughty();
}

unsafe fn bar() {
  foo.naughty();
}

fn main() {
  bar();
}