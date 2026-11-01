struct Foo;

impl Foo {
    const A_CONST: usize = 1;

    fn foo() -> usize {
        A_CONST //~ ERROR cannot find value `A_CONST` in this scope
    }
}

fn main() {}
