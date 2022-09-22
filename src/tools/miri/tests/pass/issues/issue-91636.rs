type BuiltIn = for<'a> fn(&str);

struct Function {
    inner: BuiltIn,
}

impl Function {
    fn new(subr: BuiltIn) -> Self {
        Self { inner: subr }
    }
}

fn dummy(_: &str) {}

fn main() {
    let func1 = Function::new(dummy);
    let func2 = Function::new(dummy);
    let inner: fn(&'static _) -> _ = func1.inner;
    assert!(inner == func2.inner);
}
