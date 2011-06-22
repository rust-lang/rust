// error-pattern:Unbound type parameter in callee
/* I'm actually not sure whether this should compile.
   But having a nice error message seems better than
   a bounds check failure (which is what was happening
   before.) */
fn hd[U](&vec[U] v) -> U {
  fn hd1(&vec[U] w) -> U {
    ret w.(0);
  }
  ret hd1(v);
}
