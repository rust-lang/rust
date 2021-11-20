struct Foo<B> {
    buffer: B
}

fn bar() {
    let Foo<Vec<u8>>  //~ ERROR expected one of `:`, `;`, `=`, `@`, or `|`, found `<`
}

fn main() {}
