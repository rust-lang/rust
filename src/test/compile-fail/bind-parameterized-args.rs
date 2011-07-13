// xfail-stage0
// error-pattern:Bind arguments with types containing parameters must be
fn main() {
  fn echo[T](int c, vec[T] x) {
  }

  let fn(vec[int]) -> () y = bind echo(42, _);

  y([1]);
}
