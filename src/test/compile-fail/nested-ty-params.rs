// xfail-stage0
// error-pattern:Attempt to use a type argument out of scope
fn hd[U](v: &vec[U]) -> U {
    fn hd1(w: &vec[U]) -> U { ret w.(0); }
    ret hd1(v);
}