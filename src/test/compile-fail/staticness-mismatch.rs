
trait foo {
    static fn bar();
}

impl int: foo {
    fn bar() {} //~ ERROR self type does not match the trait method's
}

fn main() {}
