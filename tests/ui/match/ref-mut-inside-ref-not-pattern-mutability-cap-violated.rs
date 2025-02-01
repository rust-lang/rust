// This test is for https://github.com/rust-lang/rust/issues/136223.
// `&mut` inside `&` should not panic by `Pattern mutability cap violdated` debug assertion.
fn main() {
  if let &Some(Some(x)) = &Some(&mut Some(0)) {}
  //~^ ERROR: cannot borrow data in a `&` reference as mutable
}
