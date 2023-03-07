// error-pattern:can't use generic parameters from outer function
fn hd<U>(v: Vec<U> ) -> U {
    fn hd1(w: [U]) -> U { return w[0]; }

    return hd1(v);
}

fn main() {}
