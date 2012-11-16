
trait foo {
    static fn bar();
}

impl int: foo {
    fn bar() {} //~ ERROR method `bar` is declared as static in its trait, but not in its impl
}

fn main() {}
