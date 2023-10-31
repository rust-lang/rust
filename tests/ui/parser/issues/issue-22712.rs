struct Foo<B> {
    buffer: B
}

fn bar() {
    let Foo<Vec<u8>> //~ ERROR generic args in patterns require the turbofish syntax
}

fn main() {}
