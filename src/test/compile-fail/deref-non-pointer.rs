// error-pattern:cannot be dereferenced
fn main() {
  alt *1 {
      _ { fail; }
  }
}