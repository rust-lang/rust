// error-pattern: calculated effect is 'unsafe'

native mod foo {
  fn naughty();
}

fn main() {
  foo.naughty();
}