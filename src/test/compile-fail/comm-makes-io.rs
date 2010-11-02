// error-pattern: calculated effect is 'impure'

fn main() {
  let chan[int] c = chan();
  c <| 10;
}