// error-pattern: calculated effect is 'io'

io fn foo() {
  let chan[int] c = chan();
  c <| 10;
}

fn main() {
  foo();
}