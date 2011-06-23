// error-pattern: Unsatisfied precondition constraint (for example, init(x
fn main() {
  auto x = @5;
  auto y <- x;
  log *x;
}