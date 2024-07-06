//@ known-bug: rust-lang/rust #127332

async fn fun() {
    enum Foo {
        A { x: u32 },
    }
    let orig = Foo::A { x: 5 };
    Foo::A { x: 6, ..orig };
}
