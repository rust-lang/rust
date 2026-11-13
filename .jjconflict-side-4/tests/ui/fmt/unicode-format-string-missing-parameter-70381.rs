// https://github.com/rust-lang/rust/issues/70381
// Test that multi-byte unicode characters with missing parameters do not ICE.

fn main() {
  println!("\r¡{}")
  //~^ ERROR 1 positional argument in format string
}
