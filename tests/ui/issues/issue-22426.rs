//@ run-pass

fn main() {
  match 42 {
    x if x < 7 => (),
    _ => ()
  }
}
