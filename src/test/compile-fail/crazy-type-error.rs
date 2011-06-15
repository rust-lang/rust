// error-pattern: mismatched types

tag t { a; }

fn f(int a) {}

fn main() {
  f(a);
}