trait A { fn foo(); }
trait B { fn foo(); }

struct AB {}

impl A for AB {
    fn foo() {}
}

impl B for AB {
    fn foo() {}
}

fn main() {
    AB::foo();  //~ ERROR E0034
}
