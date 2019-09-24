// build-pass (FIXME(62277): could be check-pass?)

// rust-lang/rust#55810: types for a binding in a match arm can be
// inferred from arms that come later in the match.

struct S;

impl S {
    fn method(&self) -> bool {
        unimplemented!()
    }
}

fn get<T>() -> T {
    unimplemented!()
}

fn main() {
    match get() {
        x if x.method() => {}
        &S => {}
    }
}
