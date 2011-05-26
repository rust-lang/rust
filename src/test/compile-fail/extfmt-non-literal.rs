// xfail-stage0
// error-pattern: literal

fn main() {
  // #fmt's first argument must be a literal.  Hopefully this
  // restriction can be eased eventually to just require a
  // compile-time constant.
  auto x = #fmt("a" + "b");
}
