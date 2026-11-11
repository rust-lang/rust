//@ run-pass
fn foo() -> impl Into<for<'a> fn(&'a ())> {
    (|_| {}) as for<'a> fn(&'a ())
}

fn main() {
    foo().into()(&());
}
