struct Foo {
    x: u32,
}
struct Bar;

fn main() {
    let x = Foo { x: 0 };
    let _ = x.foo; //~ ERROR E0609

    let y = Bar;
    y.1; //~ ERROR E0609
}
