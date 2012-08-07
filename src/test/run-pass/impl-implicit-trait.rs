enum option_<T> {
    none_,
    some_(T),
}

impl<T> option_<T> {
    fn foo() -> bool { true }
}

enum option__ {
    none__,
    some__(int)
}

impl option__ {
    fn foo() -> bool { true }
}

fn main() {
}