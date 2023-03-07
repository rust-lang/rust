pub enum Foo {
    A,
    B(isize),
    C { a: isize },
}

impl Foo {
    pub fn foo() {}
    pub fn bar(&self) {}
}
