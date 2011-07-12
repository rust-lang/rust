// xfail-stage0
// error-pattern:Attempt to use a type argument out of scope
fn hd[U](&vec[U] v) -> U {
  fn hd1(&vec[U] w) -> U {
    ret w.(0);
  }
  ret hd1(v);
}
