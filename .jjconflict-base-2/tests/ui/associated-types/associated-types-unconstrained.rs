// Check that an associated type cannot be bound in an expression path.

trait Foo {
    type A;
    fn bar() -> isize;
}

impl Foo for isize {
    type A = usize;
    fn bar() -> isize { 42 }
}

pub fn main() {
    let x: isize = Foo::bar();
    //~^ ERROR E0790
}
