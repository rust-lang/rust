struct Foo {
    bar: String,
}

impl Foo {
    pub fn new(bar: impl ToString) -> Self {
        Self {
            bar: bar.into(), //~ ERROR the trait bound `String: From<impl ToString>` is not satisfied
        }
    }
}

fn main() {}
