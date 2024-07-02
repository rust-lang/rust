struct Foo {
    field: u32,
}

impl Foo {
    fn field(&self) -> u32 {
        self.field
    }

    fn new() -> Foo {
        field; //~ ERROR cannot find value `field`
        Foo { field } //~ ERROR cannot find value `field`
    }
    fn clone(&self) -> Foo {
        Foo { field } //~ ERROR cannot find value `field`
    }
}
fn main() {}
