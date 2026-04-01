struct Foo {
    field: u32,
}

impl Foo {
    fn field(&self) -> u32 {
        self.field
    }

    fn new() -> Foo {
        field; //~ ERROR cannot find value `field` in this scope
        Foo { field } //~ ERROR cannot find value `field` in this scope
    }
    fn clone(&self) -> Foo {
        Foo { field } //~ ERROR cannot find value `field` in this scope
    }
}
fn main() {}
