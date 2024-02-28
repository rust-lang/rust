struct Foo {
    bar: String,
}

impl Foo {
    pub fn new(bar: impl ToString) -> Self {
        Self {
            bar: bar.into(), //~ ERROR trait `From<impl ToString>` is not implemented for `String`
        }
    }
}

fn main() {}
