struct Foo {
    x: u32,
    y: u32,
}
fn foo() -> Foo {
    let x = 1;
    let y = 2;

    Foo { x: x, y: y }
}
