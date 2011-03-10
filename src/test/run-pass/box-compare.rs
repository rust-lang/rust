fn main() {
  check (@1 < @3);
  check (@@"hello " > @@"hello");
  check (@@@"hello" != @@@"there");
}