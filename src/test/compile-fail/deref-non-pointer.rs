// error-pattern:cannot be dereferenced
fn main() {
  match *1 {
      _ => { fail; }
  }
}