// run-pass
// pretty-expanded FIXME #23616

fn main() {
  match 42 {
    x if x < 7 => (),
    _ => ()
  }
}
