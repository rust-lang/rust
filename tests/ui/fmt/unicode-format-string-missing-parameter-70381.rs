// https://github.com/rust-lang/rust/issues/70381
// Test that multi-byte unicode characters with missing parameters do not ICE.

fn main() {
  println!("¡{}")
  //~^ ERROR 1 positional argument in format string
}
