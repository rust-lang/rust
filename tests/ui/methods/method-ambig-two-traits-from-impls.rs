trait A { fn foo(self); }
trait B { fn foo(self); }

struct AB {}

impl A for AB {
    fn foo(self) {}
}

impl B for AB {
    fn foo(self) {}
}

fn main() {
    AB {}.foo();  //~ ERROR E0034
}
