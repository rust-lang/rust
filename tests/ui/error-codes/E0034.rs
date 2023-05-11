struct Test;

trait Trait1 {
    fn foo();
}

trait Trait2 {
    fn foo();
}

impl Trait1 for Test {
    fn foo() {}
}

impl Trait2 for Test {
    fn foo() {}
}

fn main() {
    Test::foo() //~ ERROR multiple applicable items in scope
}
