// error-pattern:attempt to use a type argument out of scope
fn hd<U>(v: ~[U]) -> U {
    fn hd1(w: [U]) -> U { return w[0]; }

    return hd1(v);
}
