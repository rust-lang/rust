struct Foo {
    x: u32,
}

impl Foo {
    fn method(&self) {}
}

fn main() {
    let f = Foo { x: 0 };
    f.method; //~ ERROR E0615
}
