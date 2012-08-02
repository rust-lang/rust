
trait foo {
    static fn bar();
}

impl of foo for int {
    fn bar() {} //~ ERROR self type does not match the trait method's
}

fn main() {}
