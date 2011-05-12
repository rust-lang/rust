// error-pattern:cyclic import

mod a {
  import b::x;
}

mod b {
  import a::x;

  fn main() {
    auto y = x;
  }
}
