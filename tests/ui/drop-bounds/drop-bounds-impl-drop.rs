//@ run-pass
#![deny(drop_bounds)]
// As a special exemption, `impl Drop` in the return position raises no error.
// This allows a convenient way to return an unnamed drop guard.
fn unnameable_type() -> impl Drop {
  struct Unnameable;
  impl Drop for Unnameable {
    fn drop(&mut self) {}
  }
  Unnameable
}
fn main() {
  let _ = unnameable_type();
}
