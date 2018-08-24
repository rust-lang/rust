trait Foo {
    type bar;
    fn bar();
}

impl Foo for () {
    type bar = ();
    fn bar() {}
}

fn main() {
    let x: <() as Foo>::bar = ();
    <()>::bar();
}
