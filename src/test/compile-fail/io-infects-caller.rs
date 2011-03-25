// xfail-stage0
// error-pattern: calculated effect is 'impure'

impure fn foo() {
  let chan[int] c = chan();
  c <| 10;
}

fn main() {
  foo();
}