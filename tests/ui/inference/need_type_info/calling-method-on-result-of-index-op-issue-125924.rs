struct Foo;

impl Foo {
    fn foo(&self) {}
}

fn _bar() {
    let x = vec![]; //~ ERROR type annotations needed for `Vec<_>`
    x[0usize].foo();
}

fn main() {
    let thing: Vec<Foo> = vec![];

    let foo = |i| {
        thing[i].foo(); //~ ERROR type annotations needed
    };
}
