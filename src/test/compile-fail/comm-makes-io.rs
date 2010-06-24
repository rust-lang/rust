// error-pattern: calculated effect is 'io'

fn main() {
  let chan[int] c = chan();
  c <| 10;
}