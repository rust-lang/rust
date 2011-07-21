fn main() {
  #macro([#apply(f,[x,...]), f(x, ...)]);

  fn add(int a, int b) -> int {
    ret a+b;
  }

  assert(#apply(add, [1, 15]) == 16);
}