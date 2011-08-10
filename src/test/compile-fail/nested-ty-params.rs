// error-pattern:Attempt to use a type argument out of scope
fn hd[U](v: &[U]) -> U {
    fn hd1(w: &[U]) -> U { ret w.(0); }

    ret hd1(v);
}
